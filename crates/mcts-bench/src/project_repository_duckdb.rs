//! DuckDB implementation of [`crate::project_repository::ProjectRepository`].

use std::sync::{Arc, Mutex};

use duckdb::{params, Connection, Transaction};

use crate::project_repository::{
    CreateExperiment, CreateProject, Experiment, Project, ProjectRepository,
    ProjectRepositoryError, UpdateExperiment, UpdateProject,
};

/// A project repository backed by a shared DuckDB connection.
#[derive(Clone)]
pub struct SharedDuckDbProjectRepository {
    connection: Arc<Mutex<Connection>>,
}

impl SharedDuckDbProjectRepository {
    pub fn new(connection: Arc<Mutex<Connection>>) -> Self {
        Self { connection }
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>, ProjectRepositoryError> {
        self.connection.lock().map_err(|_| {
            ProjectRepositoryError::Storage("benchmark database mutex poisoned".into())
        })
    }
}

impl ProjectRepository for SharedDuckDbProjectRepository {
    fn list_active_projects(&self) -> Result<Vec<Project>, ProjectRepositoryError> {
        let connection = self.lock()?;
        list_active_projects(&connection)
    }

    fn create_project(&self, request: CreateProject) -> Result<Project, ProjectRepositoryError> {
        let connection = self.lock()?;
        create_project(&connection, request)
    }

    fn load_project(&self, project_id: &str) -> Result<Project, ProjectRepositoryError> {
        let connection = self.lock()?;
        load_project(&connection, project_id)
    }

    fn update_project(&self, request: UpdateProject) -> Result<Project, ProjectRepositoryError> {
        let mut connection = self.lock()?;
        update_project(&mut connection, request)
    }

    fn list_experiments(
        &self,
        project_id: &str,
    ) -> Result<Vec<Experiment>, ProjectRepositoryError> {
        let connection = self.lock()?;
        list_experiments(&connection, project_id)
    }

    fn create_experiment(
        &self,
        request: CreateExperiment,
    ) -> Result<Experiment, ProjectRepositoryError> {
        let mut connection = self.lock()?;
        create_experiment(&mut connection, request)
    }

    fn load_experiment(&self, experiment_id: &str) -> Result<Experiment, ProjectRepositoryError> {
        let connection = self.lock()?;
        load_experiment(&connection, experiment_id)
    }

    fn update_experiment(
        &self,
        request: UpdateExperiment,
    ) -> Result<Experiment, ProjectRepositoryError> {
        let mut connection = self.lock()?;
        update_experiment(&mut connection, request)
    }
}

fn list_active_projects(connection: &Connection) -> Result<Vec<Project>, ProjectRepositoryError> {
    let mut statement = connection
        .prepare(
            "SELECT project_id, name, description, archived, CAST(created_at AS TEXT), CAST(updated_at AS TEXT) \
             FROM projects WHERE archived = false ORDER BY name",
        )
        .map_err(storage)?;
    statement
        .query_map([], project_from_row)
        .map_err(storage)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(storage)
}

fn create_project(
    connection: &Connection,
    request: CreateProject,
) -> Result<Project, ProjectRepositoryError> {
    let existing: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM projects WHERE name = ?1 AND archived = false",
            params![request.name],
            |row| row.get(0),
        )
        .map_err(storage)?;
    if existing > 0 {
        return Err(ProjectRepositoryError::Conflict);
    }
    connection
        .execute(
            "INSERT INTO projects (project_id, name, description, archived, created_at, updated_at) \
             VALUES (?1, ?2, ?3, false, ?4, ?4)",
            params![request.project_id, request.name, request.description, request.created_at],
        )
        .map_err(storage)?;
    load_project(connection, &request.project_id)
}

fn load_project(
    connection: &Connection,
    project_id: &str,
) -> Result<Project, ProjectRepositoryError> {
    connection
        .query_row(
            "SELECT project_id, name, description, archived, CAST(created_at AS TEXT), CAST(updated_at AS TEXT) \
             FROM projects WHERE project_id = ?1",
            params![project_id],
            project_from_row,
        )
        .map_err(not_found_or_storage)
}

fn update_project(
    connection: &mut Connection,
    request: UpdateProject,
) -> Result<Project, ProjectRepositoryError> {
    let transaction = connection.unchecked_transaction().map_err(storage)?;
    let current = transaction
        .query_row(
            "SELECT name, description, archived FROM projects WHERE project_id = ?1",
            params![request.project_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, bool>(2)?,
                ))
            },
        )
        .map_err(not_found_or_storage)?;
    let name = request.name.unwrap_or(current.0);
    let description = request.description.unwrap_or(current.1);
    let archived = request.archived.unwrap_or(current.2);
    let duplicate: i64 = transaction
        .query_row(
            "SELECT COUNT(*) FROM projects WHERE project_id <> ?1 AND name = ?2 AND archived = false",
            params![request.project_id, name],
            |row| row.get(0),
        )
        .map_err(storage)?;
    if duplicate > 0 && !archived {
        return Err(ProjectRepositoryError::Conflict);
    }
    transaction
        .execute(
            "UPDATE projects SET name = ?1, description = ?2, archived = ?3, updated_at = ?4 WHERE project_id = ?5",
            params![name, description, archived, request.updated_at, request.project_id],
        )
        .map_err(storage)?;
    let project = transaction
        .query_row(
            "SELECT project_id, name, description, archived, CAST(created_at AS TEXT), CAST(updated_at AS TEXT) \
             FROM projects WHERE project_id = ?1",
            params![request.project_id],
            project_from_row,
        )
        .map_err(storage)?;
    transaction.commit().map_err(storage)?;
    Ok(project)
}

fn list_experiments(
    connection: &Connection,
    project_id: &str,
) -> Result<Vec<Experiment>, ProjectRepositoryError> {
    load_project(connection, project_id)?;
    let mut statement = connection
        .prepare(
            "SELECT experiment_id, project_id, name, description, CAST(spec AS TEXT), \
             CAST(created_at AS TEXT), CAST(updated_at AS TEXT) \
             FROM experiments WHERE project_id = ?1 ORDER BY name",
        )
        .map_err(storage)?;
    statement
        .query_map(params![project_id], experiment_from_row)
        .map_err(storage)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(storage)
}

fn create_experiment(
    connection: &mut Connection,
    request: CreateExperiment,
) -> Result<Experiment, ProjectRepositoryError> {
    let transaction = connection.unchecked_transaction().map_err(storage)?;
    let parent: i64 = transaction
        .query_row(
            "SELECT COUNT(*) FROM projects WHERE project_id = ?1",
            params![request.project_id],
            |row| row.get(0),
        )
        .map_err(storage)?;
    if parent == 0 {
        return Err(ProjectRepositoryError::NotFound);
    }
    let duplicate: i64 = transaction
        .query_row(
            "SELECT COUNT(*) FROM experiments WHERE project_id = ?1 AND name = ?2",
            params![request.project_id, request.name],
            |row| row.get(0),
        )
        .map_err(storage)?;
    if duplicate > 0 {
        return Err(ProjectRepositoryError::Conflict);
    }
    transaction
        .execute(
            "INSERT INTO experiments (experiment_id, project_id, name, description, spec, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
            params![request.experiment_id, request.project_id, request.name, request.description, request.spec_json, request.created_at],
        )
        .map_err(storage)?;
    let experiment = load_experiment_from_transaction(&transaction, &request.experiment_id)?;
    transaction.commit().map_err(storage)?;
    Ok(experiment)
}

fn load_experiment(
    connection: &Connection,
    experiment_id: &str,
) -> Result<Experiment, ProjectRepositoryError> {
    connection
        .query_row(
            "SELECT experiment_id, project_id, name, description, CAST(spec AS TEXT), \
             CAST(created_at AS TEXT), CAST(updated_at AS TEXT) \
             FROM experiments WHERE experiment_id = ?1",
            params![experiment_id],
            experiment_from_row,
        )
        .map_err(not_found_or_storage)
}

fn load_experiment_from_transaction(
    transaction: &Transaction<'_>,
    experiment_id: &str,
) -> Result<Experiment, ProjectRepositoryError> {
    transaction
        .query_row(
            "SELECT experiment_id, project_id, name, description, CAST(spec AS TEXT), \
             CAST(created_at AS TEXT), CAST(updated_at AS TEXT) \
             FROM experiments WHERE experiment_id = ?1",
            params![experiment_id],
            experiment_from_row,
        )
        .map_err(not_found_or_storage)
}

fn update_experiment(
    connection: &mut Connection,
    request: UpdateExperiment,
) -> Result<Experiment, ProjectRepositoryError> {
    let transaction = connection.unchecked_transaction().map_err(storage)?;
    let project_id: String = transaction
        .query_row(
            "SELECT project_id FROM experiments WHERE experiment_id = ?1",
            params![request.experiment_id],
            |row| row.get(0),
        )
        .map_err(not_found_or_storage)?;
    let duplicate: i64 = transaction
        .query_row(
            "SELECT COUNT(*) FROM experiments WHERE project_id = ?1 AND experiment_id <> ?2 AND name = ?3",
            params![project_id, request.experiment_id, request.name],
            |row| row.get(0),
        )
        .map_err(storage)?;
    if duplicate > 0 {
        return Err(ProjectRepositoryError::Conflict);
    }
    transaction
        .execute(
            "UPDATE experiments SET name = ?1, description = ?2, spec = ?3, updated_at = ?4 WHERE experiment_id = ?5",
            params![request.name, request.description, request.spec_json, request.updated_at, request.experiment_id],
        )
        .map_err(storage)?;
    let experiment = load_experiment_from_transaction(&transaction, &request.experiment_id)?;
    transaction.commit().map_err(storage)?;
    Ok(experiment)
}

fn project_from_row(row: &duckdb::Row<'_>) -> duckdb::Result<Project> {
    Ok(Project {
        project_id: row.get(0)?,
        name: row.get(1)?,
        description: row.get(2)?,
        archived: row.get(3)?,
        created_at: row.get(4)?,
        updated_at: row.get(5)?,
    })
}

fn experiment_from_row(row: &duckdb::Row<'_>) -> duckdb::Result<Experiment> {
    Ok(Experiment {
        experiment_id: row.get(0)?,
        project_id: row.get(1)?,
        name: row.get(2)?,
        description: row.get(3)?,
        spec_json: row.get(4)?,
        created_at: row.get(5)?,
        updated_at: row.get(6)?,
    })
}

fn storage(error: duckdb::Error) -> ProjectRepositoryError {
    ProjectRepositoryError::Storage(error.to_string())
}

fn not_found_or_storage(error: duckdb::Error) -> ProjectRepositoryError {
    match error {
        duckdb::Error::QueryReturnedNoRows => ProjectRepositoryError::NotFound,
        error => storage(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_projects_and_reports_active_name_conflicts() {
        let connection = Arc::new(Mutex::new(Connection::open_in_memory().unwrap()));
        crate::schema::ensure_schema(&connection.lock().unwrap()).unwrap();
        let repository = SharedDuckDbProjectRepository::new(connection);
        repository
            .create_project(CreateProject {
                project_id: "first".into(),
                name: "First".into(),
                description: String::new(),
                created_at: "2026-01-01T00:00:00Z".into(),
            })
            .unwrap();
        repository
            .create_project(CreateProject {
                project_id: "second".into(),
                name: "Second".into(),
                description: String::new(),
                created_at: "2026-01-01T00:00:00Z".into(),
            })
            .unwrap();

        assert!(matches!(
            repository.update_project(UpdateProject {
                project_id: "second".into(),
                name: Some("First".into()),
                description: None,
                archived: None,
                updated_at: "2026-01-02T00:00:00Z".into(),
            }),
            Err(ProjectRepositoryError::Conflict)
        ));
    }
}
