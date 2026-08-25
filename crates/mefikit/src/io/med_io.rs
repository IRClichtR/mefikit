use crate::mesh::ConnectivityView;
use crate::mesh::ElementBlockView;
use crate::mesh::ElementType;
use crate::mesh::Regularity;
use crate::mesh::UMeshView;

use hdf5_metno::types::VarLenAscii;
use hdf5_metno::{File, Group};
use ndarray::prelude::*;

impl ElementType {
    pub fn med_name(self) -> &'static str {
        match self {
            ElementType::VERTEX => "PO1",
            ElementType::SEG2 => "SE2",
            ElementType::SEG3 => "SE3",

            ElementType::TRI3 => "TR3",
            ElementType::TRI6 => "TR6",
            ElementType::TRI7 => "TR7",

            ElementType::QUAD4 => "QU4",
            ElementType::QUAD8 => "QU8",
            ElementType::QUAD9 => "QU9",

            ElementType::TET4 => "TE4",
            ElementType::TET10 => "T10",

            ElementType::HEX8 => "HE8",
            // ElementType::HEX21     => "H20",
            // ElementType::Hexa27     => "H27",

            // ElementType::Pyramid5   => "PY5",
            // ElementType::Pyramid13  => "P13",

            // ElementType::Wedge6     => "PE6",
            // ElementType::Wedge15    => "P15",
            ElementType::PGON => "POG",
            _ => todo!(),
        }
    }
}

pub fn write(
    path: impl AsRef<std::path::Path>,
    mesh: &UMeshView,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = hdf5_metno::File::create(path)?;

    write_general_info(&file)?;
    write_mesh(&file, mesh)?;

    Ok(())
}

fn write_general_info(file: &File) -> hdf5_metno::Result<()> {
    let info = file.create_group("INFOS_GENERALES")?;

    write_scalar_attr(&info, "MAJ", 4i32)?;
    write_scalar_attr(&info, "MIN", 1i32)?;
    write_scalar_attr(&info, "REL", 0i32)?;

    Ok(())
}

fn write_scalar_attr<T: hdf5_metno::H5Type>(
    obj: &hdf5_metno::Location,
    name: &str,
    value: T,
) -> hdf5_metno::Result<()> {
    obj.new_attr::<T>().create(name)?.write_scalar(&value)
}

fn write_mesh(file: &File, mesh: &UMeshView) -> hdf5_metno::Result<()> {
    let ensemble = file.create_group("ENS_MAA")?;
    let med_mesh = ensemble.create_group("mesh")?;

    let dim = mesh.coords().shape()[1] as i32;

    write_scalar_attr(&med_mesh, "DIM", dim)?;
    write_scalar_attr(&med_mesh, "ESP", dim)?;
    write_scalar_attr(&med_mesh, "REP", 0i32)?;
    write_scalar_attr(&med_mesh, "SRT", 1i32)?;
    write_scalar_attr(&med_mesh, "TYP", 0i32)?;

    let timestep = med_mesh.create_group("-0000000000000000001-0000000000000000001")?;

    write_scalar_attr(&timestep, "CGT", 1i32)?;
    write_scalar_attr(&timestep, "NDT", -1i32)?;
    write_scalar_attr(&timestep, "NOR", -1i32)?;
    write_scalar_attr(&timestep, "PDT", -1.0f64)?;

    write_nodes(&timestep, &mesh.coords())?;

    let mai = timestep.create_group("MAI")?;

    write_scalar_attr(&mai, "CGT", 1i32)?;

    for (et, block) in mesh.blocks() {
        write_block(&mai, et, &block.view())?;
    }

    Ok(())
}

fn write_nodes(timestep: &Group, coords: &ArrayView2<f64>) -> hdf5_metno::Result<()> {
    let noe = timestep.create_group("NOE")?;

    write_scalar_attr(&noe, "CGT", 1i32)?;
    write_scalar_attr(&noe, "CGS", 1i32)?;

    let coo_data: Vec<f64> = {
        let (n, dim) = coords.dim();
        let mut out = Vec::with_capacity(n * dim);

        for d in 0..dim {
            for n in 0..n {
                out.push(coords[[n, d]]);
            }
        }

        out
    };

    let coo = noe
        .new_dataset_builder()
        .with_data(&coo_data)
        .create("COO")?;

    write_scalar_attr(&coo, "CGT", 1i32)?;
    write_scalar_attr(&coo, "NBR", coords.nrows() as i64)?;

    Ok(())
}

impl ElementType {
    fn med_permutation(self) -> Option<&'static [usize]> {
        match self {
            ElementType::TET4 => Some(&[0, 1, 3, 2]),

            ElementType::HEX8 => Some(&[4, 5, 6, 7, 0, 1, 2, 3]),

            ElementType::TET10 => Some(&[0, 1, 3, 2, 4, 8, 7, 6, 5, 9]),

            _ => None,
        }
    }
}

fn reorder_connectivity(conn: &ArrayView2<usize>, permutation: Option<&[usize]>) -> Array2<u64> {
    let Some(p) = permutation else {
        return conn.mapv(|x| x as u64);
    };

    let mut out = Array2::<u64>::zeros((conn.nrows(), p.len()));

    for (new_j, &old_j) in p.iter().enumerate() {
        out.column_mut(new_j)
            .assign(&conn.column(old_j).mapv(|x| x as u64));
    }

    out
}

fn write_regular(
    mai: &Group,
    element_type: ElementType,
    regular: &ElementBlockView,
) -> hdf5_metno::Result<()> {
    let conn = match &regular.connectivity {
        ConnectivityView::Regular(pc) => pc,
        _ => panic!("A Regular block must contain a regular connectivity."),
    };
    let med_type = element_type.med_name();

    let group = mai.create_group(med_type)?;

    write_scalar_attr(&group, "CGT", 1i32)?;
    write_scalar_attr(&group, "CGS", 1i32)?;
    write_bytes_attr(&group, "PFL", b"MED_NO_PROFILE_INTERNAL")?;

    let med_conn = reorder_connectivity(conn, element_type.med_permutation());

    let mut flat = Vec::with_capacity(med_conn.len());

    // MED expects column-major ordering.
    for j in 0..med_conn.ncols() {
        for i in 0..med_conn.nrows() {
            flat.push(med_conn[[i, j]] + 1);
        }
    }

    let nod = group.new_dataset_builder().with_data(&flat).create("NOD")?;

    write_scalar_attr(&nod, "CGT", 1i32)?;
    write_scalar_attr(&nod, "NBR", med_conn.nrows() as i64)?;

    Ok(())
}

fn write_bytes_attr(group: &hdf5_metno::Group, name: &str, value: &[u8]) -> hdf5_metno::Result<()> {
    let value =
        VarLenAscii::from_ascii(value).map_err(|e| hdf5_metno::Error::Internal(e.to_string()))?;

    group
        .new_attr::<VarLenAscii>()
        .create(name)?
        .write_scalar(&value)?;

    Ok(())
}

fn write_polygon(mai: &Group, poly: &ElementBlockView) -> hdf5_metno::Result<()> {
    let poly_conn = match &poly.connectivity {
        ConnectivityView::Poly(pc) => pc,
        _ => panic!("A PGON block must contain a poly connectivity."),
    };
    let group = mai.create_group("POG")?;

    write_scalar_attr(&group, "CGT", 1i32)?;
    write_scalar_attr(&group, "CGS", 1i32)?;

    // MED NOD is 1-based.
    let nod: Vec<u64> = poly_conn.data.iter().map(|&x| x as u64 + 1).collect();

    let nod_ds = group.new_dataset_builder().with_data(&nod).create("NOD")?;

    write_scalar_attr(&nod_ds, "CGT", 1i32)?;
    write_scalar_attr(&nod_ds, "NBR", nod.len() as i64)?;

    // Mefikit:
    //
    // offset = [3, 7, 10]
    //
    // MED:
    //
    // INN = [1, 4, 8, 11]
    //
    let mut inn = Vec::with_capacity(poly_conn.offsets.len() + 1);

    inn.push(1u64);

    for &end in poly_conn.offsets.iter() {
        inn.push(end as u64 + 1);
    }

    let inn_ds = group.new_dataset_builder().with_data(&inn).create("INN")?;

    write_scalar_attr(&inn_ds, "CGT", 1i32)?;
    write_scalar_attr(&inn_ds, "NBR", inn.len() as i64)?;

    Ok(())
}

fn write_block(mai: &Group, et: &ElementType, block: &ElementBlockView) -> hdf5_metno::Result<()> {
    match et.regularity() {
        Regularity::Regular => {
            write_regular(mai, *et, block)?;
        }

        Regularity::Poly => match et {
            ElementType::PGON => {
                write_polygon(mai, block)?;
            }

            _ => {
                return Err(hdf5_metno::Error::Internal(format!(
                    "PolyConnectivity is not supported \
                                for {:?}",
                    et
                )));
            }
        },
    }

    Ok(())
}
