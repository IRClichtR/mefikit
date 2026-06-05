use crate::mesh::{UMesh, UMeshView};
use std::path::Path;

pub mod error;
mod hdfvtk_io;
mod serde_io;
pub mod vtk_io;
// mod med; // for later
// mod cgns; // for later

pub use error::{IOError, HdfVtkError};

pub fn read(path: &Path) -> Result<UMesh, IOError> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        "json" => serde_io::read_json(path),
        "yaml" | "yml" => serde_io::read_yaml(path),
        "vtk" | "vtu" => vtk_io::read(path),
        "vtkhdf" | "h5" | "hdf5" => hdfvtk_io::read(path),
        _ => Err(IOError::UnsupportedExtension(format!("{path:?}"))),
    }
}

pub fn write(path: &Path, mesh: UMeshView) -> Result<(), IOError> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        "json" => serde_io::write_json(path, mesh),
        "yaml" | "yml" => serde_io::write_yaml(path, mesh),
        "vtk" | "vtu" => vtk_io::write(path, mesh),
        "vtkhdf" | "h5" | "hdf5" => hdfvtk_io::write(path, mesh),
        _ => Err(IOError::UnsupportedExtension(format!("{path:?}"))),
    }
}
