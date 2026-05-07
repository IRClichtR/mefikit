use std::path::Path;
use hdf5_metno::{File, Group};
use ndarray::{Array1, Array2};
use crate::mesh::{ElementType, UMesh};

// TODO: Enum CgnsDenominations
// cgns to ElType 


fn read_string_data(group: &Group) -> Result<String, Box<dyn std::error::Error>> {
    let s: String = group
        .dataset(" data")?
        .as_reader()
        .read_1d::<i8>()?
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as u8 as char)
        .collect();
    Ok(s.trim().to_string())
}

fn cgns_label(group: &Group) -> Result<String, Box<dyn std::error::Error>> {
    let label: String = group 
        .attr(" label")?
        .as_reader()
        .read_1d::<i8>()?
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as u8 as char)
        .collect();

    Ok(label.trim().to_string())
}

fn find_first_child_with_label(group: &Group, label: &str) -> Result<Group, Box<dyn std::error::Error>> {
    for name in group.member_names()? {
        let Ok(child) = group.group(&name) else { continue };
        let Ok(child_label) = cgns_label(&child) else { continue };
        if child_label == label {
            return Ok(child)
        }
    }
    Err(format!("no child with label '{label}' in {}", group.name()).into())
}

fn find_children_with_label(group: &Group, label: &str) -> Result<Vec<Group>, Box<dyn std::error::Error>> {
    let mut children: Vec<Group> = Vec::new();

    for name in group.member_names()? {
        let child = group.group(name.as_str())?;
        let Ok(child_label) = cgns_label(&child) else { continue };
        if child_label == label {
            children.push(child);
        }
    }

    Ok(children)
}

fn read_coords(zone: &Group) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let gc = find_first_child_with_label(zone, "GridCoordinates_t")?;
    let x: Array1<f64> = gc.dataset("CoordinateX")?.as_reader().read_1d::<f64>()?;
    let y: Array1<f64> = gc.dataset("CoordinateY")?.as_reader().read_1d::<f64>()?;
    let z: Array1<f64> = gc.dataset("CoordinateZ")?.as_reader().read_1d::<f64>()?;
    let n = x.len();
    let mut coords = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        coords[[i, 0]] = x[i];
        coords[[i, 1]] = y[i];
        coords[[i, 2]] = z[i];
    }
    Ok(coords)
}

pub fn read(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    let file =  File::open(path)?;
    let base = find_first_child_with_label(&file.as_group()?, "CGNSBase_t")?;
    let zone = find_first_child_with_label(&base, "Zone_t")?;
    
    // validate zone type
    let z_type = read_string_data(&find_first_child_with_label(&zone, "ZoneType_t")?)?;
    if z_type != "Unstructured" {
        return Err(format!("unsupported zone type: {z_type}").into());
    }

    // Build mesh with coords
    let coords = read_coords(&zone)?;
    let mut mesh: UMesh = UMesh::new(coords.into());

    // read elements and push to Mesh
    // tag elements with family name
    // ?
    Ok(mesh)
}

pub fn write() {}