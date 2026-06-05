use thiserror::Error;

#[derive(Error, Debug)]
pub enum IOError {
    #[error("Unsupported file extension: {0:?}")]
    UnsupportedExtension(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("HDF5 error: {0}")]
    HDF5(#[from] hdf5_metno::Error),
    #[error("Shape error: {0}")]
    NdArray(#[from] ndarray::ShapeError),
    #[cfg(feature = "io")]
    #[error("VTK error: {0}")]
    Vtk(#[from] vtkio::Error),
    #[error("HdfVtk error: {0}")]
    HdfVtk(#[from] HdfVtkError),
}

#[derive(Error, Debug)]
pub enum HdfVtkError {
    #[error("Not a VTKHDF file")]
    NotVTKHDF,
    #[error("Unexpected string type: {0:?}")]
    UnexpectedStringType(String),
    #[error("Unsupported ElementType: {0:?}")]
    UnsupportedElementType(String),
    #[error("VTKHDF group not found in {0}")]
    VTKHDFGroupNotFound(String),
} 