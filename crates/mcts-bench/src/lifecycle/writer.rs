use super::*;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct LifecycleWriter {
    path: PathBuf,
    file: File,
    attempt_id: String,
    launch_nonce: String,
    next_sequence: u64,
    child: bool,
    terminal: bool,
    closed: bool,
    poisoned: bool,
    #[cfg(test)]
    fail_next_append: bool,
}
impl LifecycleWriter {
    pub fn create(
        path: impl AsRef<Path>,
        manifest: WrapperManifest,
        launch_nonce: impl Into<String>,
        timestamp: impl Into<String>,
    ) -> Result<Self, LifecycleError> {
        let path = path.as_ref().to_path_buf();
        let nonce = launch_nonce.into();
        let timestamp = timestamp.into();
        named(&nonce, "launch_nonce")
            .and_then(|_| named(&timestamp, "timestamp"))
            .and_then(|_| validate_manifest(&manifest))
            .map_err(|e| invalid(&path, None, None, e))?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|source| {
                if source.kind() == std::io::ErrorKind::AlreadyExists {
                    LifecycleError::Conflict { path: path.clone() }
                } else {
                    LifecycleError::Io {
                        path: path.clone(),
                        operation: "create",
                        source,
                    }
                }
            })?;
        let mut w = Self {
            path,
            file,
            attempt_id: manifest.attempt_id.clone(),
            launch_nonce: nonce,
            next_sequence: 0,
            child: false,
            terminal: false,
            closed: false,
            poisoned: false,
            #[cfg(test)]
            fail_next_append: false,
        };
        if let Err(e) = w.append(LifecyclePayload::WrapperStarted(manifest), timestamp, true) {
            w.poisoned = true;
            return Err(e);
        }
        Ok(w)
    }
    fn append(
        &mut self,
        payload: LifecyclePayload,
        timestamp: String,
        durable: bool,
    ) -> Result<(), LifecycleError> {
        if self.poisoned {
            return Err(LifecycleError::Poisoned {
                path: self.path.clone(),
            });
        }
        #[cfg(test)]
        if self.fail_next_append {
            self.fail_next_append = false;
            self.poisoned = true;
            return Err(LifecycleError::Io {
                path: self.path.clone(),
                operation: "write",
                source: std::io::Error::other("injected failure"),
            });
        }
        let r = LifecycleRecord {
            schema_version: 1,
            sequence: self.next_sequence,
            attempt_id: self.attempt_id.clone(),
            launch_nonce: self.launch_nonce.clone(),
            timestamp,
            payload,
        };
        let result = named(&r.timestamp, "timestamp")
            .and_then(|_| validate_payload(&r.payload))
            .map_err(|e| invalid(&self.path, None, Some(r.sequence), e))
            .and_then(|_| {
                serde_json::to_writer(&mut self.file, &r).map_err(|e| LifecycleError::Io {
                    path: self.path.clone(),
                    operation: "write",
                    source: std::io::Error::other(e),
                })
            })
            .and_then(|_| {
                self.file
                    .write_all(b"\n")
                    .map_err(|source| LifecycleError::Io {
                        path: self.path.clone(),
                        operation: "write",
                        source,
                    })
            })
            .and_then(|_| {
                self.file.flush().map_err(|source| LifecycleError::Io {
                    path: self.path.clone(),
                    operation: "flush",
                    source,
                })
            })
            .and_then(|_| {
                if durable {
                    self.file.sync_data().map_err(|source| LifecycleError::Io {
                        path: self.path.clone(),
                        operation: "sync",
                        source,
                    })
                } else {
                    Ok(())
                }
            });
        if let Err(e) = result {
            self.poisoned = true;
            return Err(e);
        }
        self.next_sequence += 1;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn fail_next_append_for_test(&mut self) {
        self.fail_next_append = true;
    }
    pub fn child_started(
        &mut self,
        pid: u64,
        timestamp: impl Into<String>,
    ) -> Result<(), LifecycleError> {
        if self.child || self.terminal || self.closed {
            return Err(invalid(
                &self.path,
                None,
                Some(self.next_sequence),
                InvalidReason::InvalidTypedRecordOrdering,
            ));
        }
        if pid == 0 {
            return Err(invalid(
                &self.path,
                None,
                Some(self.next_sequence),
                InvalidReason::InvalidNamedField { field: "child_pid" },
            ));
        }
        let result = self.append(
            LifecyclePayload::ChildStarted { child_pid: pid },
            timestamp.into(),
            false,
        );
        if result.is_ok() {
            self.child = true
        }
        result
    }
    pub fn child_spawn_failed(
        &mut self,
        stage: impl Into<String>,
        error: impl Into<String>,
        timestamp: impl Into<String>,
    ) -> Result<(), LifecycleError> {
        if self.child || self.terminal || self.closed {
            return Err(invalid(
                &self.path,
                None,
                Some(self.next_sequence),
                InvalidReason::InvalidTypedRecordOrdering,
            ));
        }
        let result = self.append(
            LifecyclePayload::ChildSpawnFailed {
                stage: stage.into(),
                error: error.into(),
            },
            timestamp.into(),
            true,
        );
        if result.is_ok() {
            self.terminal = true
        }
        result
    }
    pub fn child_exited(
        &mut self,
        outcome: ExitEvidence,
        timestamp: impl Into<String>,
    ) -> Result<(), LifecycleError> {
        if !self.child || self.terminal || self.closed {
            return Err(invalid(
                &self.path,
                None,
                Some(self.next_sequence),
                InvalidReason::InvalidTypedRecordOrdering,
            ));
        }
        let result = self.append(
            LifecyclePayload::ChildExited { outcome },
            timestamp.into(),
            true,
        );
        if result.is_ok() {
            self.terminal = true
        }
        result
    }
    pub fn outputs_closed(
        &mut self,
        outputs: Vec<OutputClosure>,
        timestamp: impl Into<String>,
    ) -> Result<(), LifecycleError> {
        if !self.terminal || self.closed {
            return Err(invalid(
                &self.path,
                None,
                Some(self.next_sequence),
                InvalidReason::InvalidTypedRecordOrdering,
            ));
        }
        let result = self.append(
            LifecyclePayload::OutputsClosed { outputs },
            timestamp.into(),
            true,
        );
        if result.is_ok() {
            self.closed = true
        }
        result
    }
}
