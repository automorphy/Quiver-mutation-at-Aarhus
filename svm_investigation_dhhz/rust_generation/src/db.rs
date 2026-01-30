use rusqlite::{params, Connection, Transaction};

pub fn init_db(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS quivers (
            id INTEGER PRIMARY KEY,
            b12 INTEGER, b13 INTEGER, b14 INTEGER,
            b23 INTEGER, b24 INTEGER, b34 INTEGER,
            label INTEGER,
            seed_type TEXT,
            seed_quiver TEXT,
            sampling_method TEXT,
            mutation_depth INTEGER,
            mutation_path TEXT,
            parent_id INTEGER,
            last_vertex INTEGER,
            max_weight INTEGER,
            nma_reason TEXT,
            canonical_vec TEXT UNIQUE,
            digraph_class TEXT,
            aligned_weights TEXT
        )",
        [],
    )?;
    Ok(())
}

pub fn insert_quiver(
    tx: &Transaction,
    canon: &[i128],
    label: i32,
    seed_type: &str,
    seed_quiver: &str,
    sampling_method: &str,
    mutation_depth: i32,
    mutation_path: &str,
    parent_id: i64,
    last_vertex: i32,
    max_weight: i128,
    nma_reason: Option<String>,
    canon_str: &str,
    digraph_class: &str,
    aligned_weights: &str,
) -> rusqlite::Result<i64> {
    tx.execute(
        "INSERT OR IGNORE INTO quivers (
            b12, b13, b14, b23, b24, b34, 
            label, seed_type, seed_quiver, sampling_method, 
            mutation_depth, mutation_path, parent_id, last_vertex, 
            max_weight, nma_reason, canonical_vec, digraph_class, aligned_weights
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
        params![
            canon[0] as i64, canon[1] as i64, canon[2] as i64,
            canon[3] as i64, canon[4] as i64, canon[5] as i64,
            label, seed_type, seed_quiver, sampling_method,
            mutation_depth, mutation_path, parent_id, last_vertex,
            max_weight as i64, nma_reason, canon_str, digraph_class, aligned_weights
        ],
    )?;
    Ok(tx.last_insert_rowid())
}
