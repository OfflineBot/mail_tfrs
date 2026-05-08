//! Tiny custom binary format for transformer weights.
//!
//! Layout:
//!   magic "TRFM" (4 bytes)
//!   version u32 (LE)
//!   <component-defined sequence of Array2/Array1 dumps>
//!
//! Each Array2: rows u32, cols u32, then rows*cols f32 LE.
//! Each Array1: len u32, then len f32 LE.

use std::io::{Read, Result, Write};
use ndarray::{Array1, Array2};

pub const MAGIC: &[u8; 4] = b"TRFM";
pub const VERSION: u32 = 2;

pub fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())
}
pub fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

pub fn write_f32<W: Write>(w: &mut W, v: f32) -> Result<()> {
    w.write_all(&v.to_le_bytes())
}
pub fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

pub fn write_array2<W: Write>(w: &mut W, a: &Array2<f32>) -> Result<()> {
    let rows = a.shape()[0] as u32;
    let cols = a.shape()[1] as u32;
    write_u32(w, rows)?;
    write_u32(w, cols)?;
    let contig = a.as_standard_layout();
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            contig.as_ptr() as *const u8,
            contig.len() * std::mem::size_of::<f32>(),
        )
    };
    w.write_all(bytes)
}

pub fn read_array2<R: Read>(r: &mut R) -> Result<Array2<f32>> {
    let rows = read_u32(r)? as usize;
    let cols = read_u32(r)? as usize;
    let mut data = vec![0f32; rows * cols];
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    };
    r.read_exact(bytes)?;
    Ok(Array2::from_shape_vec((rows, cols), data).unwrap())
}

pub fn write_array1<W: Write>(w: &mut W, a: &Array1<f32>) -> Result<()> {
    write_u32(w, a.len() as u32)?;
    let contig = a.as_standard_layout();
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            contig.as_ptr() as *const u8,
            contig.len() * std::mem::size_of::<f32>(),
        )
    };
    w.write_all(bytes)
}

pub fn read_array1<R: Read>(r: &mut R) -> Result<Array1<f32>> {
    let n = read_u32(r)? as usize;
    let mut data = vec![0f32; n];
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    };
    r.read_exact(bytes)?;
    Ok(Array1::from_vec(data))
}

pub fn write_string<W: Write>(w: &mut W, s: &str) -> Result<()> {
    let bytes = s.as_bytes();
    write_u32(w, bytes.len() as u32)?;
    w.write_all(bytes)
}

pub fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let n = read_u32(r)? as usize;
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("utf8: {e}"))
    })
}

pub fn write_strings<W: Write>(w: &mut W, items: &[String]) -> Result<()> {
    write_u32(w, items.len() as u32)?;
    for s in items {
        write_string(w, s)?;
    }
    Ok(())
}

pub fn read_strings<R: Read>(r: &mut R) -> Result<Vec<String>> {
    let n = read_u32(r)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(read_string(r)?);
    }
    Ok(out)
}

pub fn write_magic<W: Write>(w: &mut W) -> Result<()> {
    w.write_all(MAGIC)?;
    write_u32(w, VERSION)
}

pub fn read_magic<R: Read>(r: &mut R) -> Result<()> {
    let mut m = [0u8; 4];
    r.read_exact(&mut m)?;
    if &m != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bad magic: {m:?}"),
        ));
    }
    let v = read_u32(r)?;
    if v != VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported version: {v}"),
        ));
    }
    Ok(())
}
