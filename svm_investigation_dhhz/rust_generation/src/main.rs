mod quiver;
mod db;

use rusqlite::Connection;
use std::collections::{HashSet, VecDeque};
use std::time::Instant;
use rand::seq::SliceRandom;
use quiver::Quiver;

// --- Rank 4 Generation Logic ---
fn run_rank4_generation() -> rusqlite::Result<()> {
    let conn = Connection::open("../data/quivers_rank4_canonical.db")?;
    db::init_db(&conn)?;

    let start_time = Instant::now();
    let mut rng = rand::thread_rng();
    let mut visited = HashSet::new();
    let mut generated = 0;
    let target = 250_000;

    // Seeds Pool
    let mut seeds = Vec::new();
    // MA Seeds (Restored for balance)
    seeds.push((Quiver::<4>::from_upper_tri(&[1, 0, 0, 1, 0, 1]), "acyclic", "A4 path"));
    seeds.push((Quiver::<4>::from_upper_tri(&[2, 0, 0, 2, 0, 2]), "acyclic", "A4 weighted"));
    seeds.push((Quiver::<4>::from_upper_tri(&[1, 1, 1, 0, 0, 0]), "acyclic", "D4 star"));
    seeds.push((Quiver::<4>::from_upper_tri(&[3, 0, 0, 0, 0, 0]), "acyclic", "A4 sparse"));
    seeds.push((Quiver::<4>::from_upper_tri(&[1, 2, 3, 0, 0, 0]), "acyclic", "Random DAG 1"));
    seeds.push((Quiver::<4>::from_upper_tri(&[0, 1, 0, 2, 0, 3]), "acyclic", "Random DAG 2"));
    seeds.push((Quiver::<4>::from_upper_tri(&[4, 0, 0, 0, 0, 0]), "acyclic", "A4 heavy"));
    
    // NMA Seeds (Expanded variety)
    seeds.push((Quiver::<4>::from_upper_tri(&[2, 2, 2, 2, 2, 2]), "torus", "Dreaded Torus")); 
    seeds.push((Quiver::<4>::from_upper_tri(&[2, 0, -2, 2, 0, -2]), "box", "2-2-2-2 Box"));
    
    // Markov seeds (Expanded)
    seeds.push((Quiver::<4>::from_upper_tri(&[3, -3, 0, 3, 0, 0]), "markov", "Markov 3-3-3 subquiver"));
    seeds.push((Quiver::<4>::from_upper_tri(&[2, -2, 0, 2, 0, 0]), "markov", "Markov 2-2-2 subquiver"));
    seeds.push((Quiver::<4>::from_upper_tri(&[4, -4, 0, 4, 0, 0]), "markov", "Markov 4-4-4 subquiver"));
    seeds.push((Quiver::<4>::from_upper_tri(&[5, -5, 0, 5, 0, 0]), "markov", "Markov 5-5-5 subquiver"));
    // Mixed Markov
    seeds.push((Quiver::<4>::from_upper_tri(&[3, -4, 0, 3, 0, 0]), "markov", "Markov 3-3-4 subquiver"));
    seeds.push((Quiver::<4>::from_upper_tri(&[3, -3, 0, 4, 0, 0]), "markov", "Markov 3-4-3 subquiver"));

    // Box variants (Box(a,b)) - Adding more to boost NMA count
    seeds.push((Quiver::<4>::from_upper_tri(&[3, 0, -3, 3, 0, -3]), "box", "3-3-3-3 Box"));
    seeds.push((Quiver::<4>::from_upper_tri(&[2, 0, -3, 2, 0, -3]), "box", "2-3-2-3 Box"));
    seeds.push((Quiver::<4>::from_upper_tri(&[2, 0, -4, 2, 0, -4]), "box", "2-4-2-4 Box"));
    seeds.push((Quiver::<4>::from_upper_tri(&[4, 0, -2, 4, 0, -2]), "box", "4-2-4-2 Box"));
    seeds.push((Quiver::<4>::from_upper_tri(&[3, 0, -4, 3, 0, -4]), "box", "3-4-3-4 Box"));

    // Heavy Torus variants
    seeds.push((Quiver::<4>::from_upper_tri(&[4, 4, 4, 4, 4, 4]), "torus", "Heavy Torus 4"));
    seeds.push((Quiver::<4>::from_upper_tri(&[3, 3, 3, 3, 3, 3]), "torus", "Heavy Torus 3"));
    seeds.push((Quiver::<4>::from_upper_tri(&[2, 2, 2, 2, 2, -2]), "custom", "Cyclic 4-cycle"));
    seeds.push((Quiver::<4>::from_upper_tri(&[3, 3, 3, 3, 3, -3]), "custom", "Cyclic 4-cycle heavy"));

    let mut queue = VecDeque::new();
    for (q, stype, name) in seeds {
        let canon = q.get_canonical_vector();
        let canon_str = format!("{:?}", canon);
        queue.push_back((q, 0, stype.to_string(), name.to_string(), -1, -1, String::new(), canon_str));
    }

    println!("\n--- Starting 250k generation for Rank 4 ---");

    let mut tx = conn.unchecked_transaction()?;
    
    while generated < target && !queue.is_empty() {
        let (q, depth, stype, sname, pid, lv, path, canon_str) = queue.pop_front().unwrap();

        if visited.contains(&canon_str) {
            continue;
        }
        visited.insert(canon_str.clone());

        let label = if stype == "acyclic" { 1 } else { 0 };
        let mw = q.max_weight();
        let nma_reason = q.nma_reason();

        let (digraph_class, aligned_weights) = q.get_digraph_class_and_aligned_weights();
        let digraph_str = format!("{:?}", digraph_class);
        let aligned_str = format!("{:?}", aligned_weights);

        let row_id = db::insert_quiver(
            &tx, &q.get_canonical_vector(), label, &stype, &sname, "bfs_shuffled", 
            depth, &path, pid, lv, mw, nma_reason, &canon_str, &digraph_str, &aligned_str
        )?;

        if row_id > 0 {
            generated += 1;
        }

        // Mutation Expansion
        if depth < 15 && mw < 1_000_000 {
            let mut neighbors: Vec<usize> = (0..4).collect();
            neighbors.shuffle(&mut rng); 
            
            for &v in &neighbors {
                if v as i32 != lv {
                    let next_q = q.mutate(v);
                    let next_canon = next_q.get_canonical_vector();
                    let next_canon_str = format!("{:?}", next_canon);
                    
                    if !visited.contains(&next_canon_str) {
                        let mut next_path = path.clone();
                        if !next_path.is_empty() { next_path.push(','); }
                        next_path.push_str(&v.to_string());
                        queue.push_back((next_q, depth + 1, stype.clone(), sname.clone(), row_id, v as i32, next_path, next_canon_str));
                    }
                }
            }
        }
        
        if generated % 25000 == 0 && row_id > 0 {
             println!("Generated {}/{}...", generated, target);
             tx.commit()?;
             tx = conn.unchecked_transaction()?;
        }
    }
    tx.commit()?;

    println!("Rank 4 Generation Complete: {}", generated);
    println!("Time taken: {:?}", start_time.elapsed());
    Ok(())
}

// --- Rank 3 Verification Logic ---
fn run_rank3_check() -> rusqlite::Result<()> {
    println!("\n--- Rank 3 Verification Run (Large Scale) ---");
    let mut conn = Connection::open("../data/quivers_rank3_verify_large.db")?;
    
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rank3_quivers (
            x INTEGER, y INTEGER, z INTEGER,
            label_via_seed INTEGER,
            label_via_theorem INTEGER,
            markov_constant INTEGER,
            is_consistent INTEGER
        )",
        [],
    )?;

    // Diverse Seeds for Rank 3
    let seeds = vec![
        (Quiver::<3>::from_upper_tri(&[1, -1, 1]), 1, "Acyclic (1,1,1)"),
        (Quiver::<3>::from_upper_tri(&[1, -2, 3]), 1, "Acyclic (1,2,3)"),
        (Quiver::<3>::from_upper_tri(&[0, 0, 0]), 1, "Acyclic (0,0,0)"),
        (Quiver::<3>::from_upper_tri(&[3, -3, 3]), 0, "Markov (3,3,3)"),
        (Quiver::<3>::from_upper_tri(&[2, -2, 2]), 0, "Cyclic (2,2,2)"),
        (Quiver::<3>::from_upper_tri(&[5, -5, 5]), 0, "Cyclic (5,5,5)"),
        (Quiver::<3>::from_upper_tri(&[4, -4, 4]), 0, "Cyclic (4,4,4)"),
    ];

    let mut queue = VecDeque::new();
    for (q, label, _desc) in seeds {
        queue.push_back((q, label));
    }

    let mut visited = HashSet::new();
    let mut count = 0;
    let target = 100_000;
    let mut mismatches = 0;

    let mut tx = conn.unchecked_transaction()?;

    while count < target && !queue.is_empty() {
        let (q, seed_label) = queue.pop_front().unwrap();
        let canon = q.get_canonical_vector();
        let canon_key = format!("{:?}", canon);

        if visited.contains(&canon_key) {
            continue;
        }
        visited.insert(canon_key);

        let paper_label = if q.is_acyclic_paper_criteria() { 1 } else { 0 };
        let consistent = if seed_label == paper_label { 1 } else { 0 };
        
        if consistent == 0 {
            mismatches += 1;
            println!("Mismatch! Quiver {:?}, Seed Label: {}, Paper Label: {}, C: {}", 
                     canon, seed_label, paper_label, q.markov_constant());
        }

        use rusqlite::params;
        tx.execute(
            "INSERT INTO rank3_quivers (x, y, z, label_via_seed, label_via_theorem, markov_constant, is_consistent)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![canon[0] as i64, canon[1] as i64, canon[2] as i64, seed_label, paper_label, q.markov_constant() as i64, consistent],
        )?;

        count += 1;

        if q.max_weight() < 1_000_000 {
            for k in 0..3 {
                queue.push_back((q.mutate(k), seed_label));
            }
        }
        
        if count % 20000 == 0 {
            println!("Verified {} rank 3 quivers...", count);
        }
    }
    tx.commit()?;

    println!("Rank 3 Verification Complete. {} entries generated.", count);
    println!("Mismatches found: {}", mismatches);
    if mismatches == 0 {
        println!("SUCCESS: All generated rank 3 quivers match the paper's classification!");
    } else {
        println!("WARNING: Found inconsistencies with the paper!");
    }

    Ok(())
}

fn run_data_analysis() -> rusqlite::Result<()> {
    let conn = Connection::open("../data/quivers_rank4_250k.db")?;
    
    println!("\n--- Dataset Analysis ---");
    
    println!("--- Label Distribution ---");
    let mut stmt = conn.prepare("SELECT label, COUNT(*) FROM quivers GROUP BY label")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, i32>(0)?, row.get::<_, i64>(1)?))
    })?;
    for row in rows {
        let (label, count) = row?;
        println!("Label {}: {}", label, count);
    }

    println!("\n--- Seed Type Distribution ---");
    let mut stmt = conn.prepare("SELECT seed_type, COUNT(*) FROM quivers GROUP BY seed_type")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })?;
    for row in rows {
        let (stype, count) = row?;
        println!("{}: {}", stype, count);
    }

    println!("\n--- Unique Canonical Forms ---");
    let count: i64 = conn.query_row("SELECT COUNT(DISTINCT canonical_vec) FROM quivers", [], |row| row.get(0))?;
    println!("Total Unique Canonical Vectors: {}", count);

    Ok(())
}

fn main() -> rusqlite::Result<()> {
    println!("=== Quiver Analysis Suite (250k Target) ===");
    run_rank3_check()?;
    run_rank4_generation()?;
    run_data_analysis()?;
    Ok(())
}
