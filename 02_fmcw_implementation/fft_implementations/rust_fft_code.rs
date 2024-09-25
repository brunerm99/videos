use rustfft::{Fft, FftDirection, num_complex::Complex, algorithm::Radix4};

fn main() {
    let fft = Radix4::new(4096, FftDirection::Forward);

    let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 4096];
    fft.process(&mut buffer);
}