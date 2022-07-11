use std::{
    fs::File,
    io::{BufRead, BufReader, Lines},
    path::Path
};

use cozy_chess::{Board, Color};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::batch::Batch;
use crate::input_features::InputFeature;

#[derive(Debug)]
pub struct AnnotatedBoard {
    board: Board,
    cp: f32,
    wdl: f32,
}

impl AnnotatedBoard {
    pub fn relative_value(&self) -> (f32, f32) {
        match self.board.side_to_move() {
            Color::White => (self.cp, self.wdl),
            Color::Black => (-self.cp, 1.0 - self.wdl),
        }
    }
}

pub struct FileReader {
    file: Lines<BufReader<File>>,
    string_buffer: Vec<String>,
    board_buffer: Vec<Option<AnnotatedBoard>>,
}

impl FileReader {
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let file = BufReader::new(file).lines();
        Ok(Self {
            file,
            string_buffer: vec![],
            board_buffer: vec![],
        })
    }

    fn fill_buffer(&mut self, chunk_size: usize) {
        self.string_buffer.clear();
        self.string_buffer.extend((&mut self.file).flat_map(Result::ok).take(chunk_size));
        self.string_buffer
            .par_iter()
            .map(|line| {
                let (board, annotation) = line.split_once(" | ")?;
                let (cp, wdl) = annotation.split_once(" | ")?;

                let cp = cp.parse::<f32>().ok()?;
                if cp.abs() > 3000.0 {
                    return None;
                }
                let wdl = wdl.parse::<f32>().ok()?;
                let board = board.parse::<Board>().ok()?;
                
                Some(AnnotatedBoard { board, cp, wdl })
            })
            .rev()
            .collect_into_vec(&mut self.board_buffer);
    }
}

impl Iterator for FileReader {
    type Item = AnnotatedBoard;

    fn next(&mut self) -> Option<Self::Item> {
        if self.board_buffer.is_empty() {
            self.fill_buffer(32_000);
        }
        while let Some(board) = self.board_buffer.pop() {
            if board.is_some() {
                return board;
            }
        }
        None
    }
}

pub fn read_batch_into<F: InputFeature>(reader: &mut FileReader, batch: &mut Batch) -> bool {
    batch.clear();
    for annotated in reader.take(batch.capacity()) {
        let (cp, wdl) = annotated.relative_value();
        let entry = batch.make_entry(cp, wdl);
        F::add_features(annotated.board, entry);
    }
    batch.capacity() == batch.len()
}
