//! Logical storage operations over benchmark projects and experiments.
//!
//! HTTP handlers depend on these application records instead of a database
//! driver, allowing another durable store or a test double to provide them.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectRepositoryError {
    NotFound,
    Conflict,
    Storage(String),
}

impl std::fmt::Display for ProjectRepositoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "benchmark project was not found"),
            Self::Conflict => write!(f, "benchmark project already exists"),
            Self::Storage(message) => write!(f, "benchmark project storage failure: {message}"),
        }
    }
}

impl std::error::Error for ProjectRepositoryError {}

#[derive(Debug, Clone)]
pub struct Project {
    pub project_id: String,
    pub name: String,
    pub description: String,
    pub archived: bool,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone)]
pub struct Experiment {
    pub experiment_id: String,
    pub project_id: String,
    pub name: String,
    pub description: String,
    /// The serialized experiment spec as it was stored.
    pub spec_json: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug)]
pub struct CreateProject {
    pub project_id: String,
    pub name: String,
    pub description: String,
    pub created_at: String,
}

#[derive(Debug)]
pub struct UpdateProject {
    pub project_id: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub archived: Option<bool>,
    pub updated_at: String,
}

#[derive(Debug)]
pub struct CreateExperiment {
    pub experiment_id: String,
    pub project_id: String,
    pub name: String,
    pub description: String,
    pub spec_json: String,
    pub created_at: String,
}

#[derive(Debug)]
pub struct UpdateExperiment {
    pub experiment_id: String,
    pub name: String,
    pub description: String,
    pub spec_json: String,
    pub updated_at: String,
}

/// Logical project and experiment storage operations.
pub trait ProjectRepository {
    fn list_active_projects(&self) -> Result<Vec<Project>, ProjectRepositoryError>;
    fn create_project(&self, request: CreateProject) -> Result<Project, ProjectRepositoryError>;
    fn load_project(&self, project_id: &str) -> Result<Project, ProjectRepositoryError>;
    fn update_project(&self, request: UpdateProject) -> Result<Project, ProjectRepositoryError>;
    fn list_experiments(&self, project_id: &str)
        -> Result<Vec<Experiment>, ProjectRepositoryError>;
    fn create_experiment(
        &self,
        request: CreateExperiment,
    ) -> Result<Experiment, ProjectRepositoryError>;
    fn load_experiment(&self, experiment_id: &str) -> Result<Experiment, ProjectRepositoryError>;
    fn update_experiment(
        &self,
        request: UpdateExperiment,
    ) -> Result<Experiment, ProjectRepositoryError>;
}
