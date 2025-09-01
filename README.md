# Bit-by-Bit-Hackathon-SCArry-monsters
Bit-by-Bit SCA Hackathon organised by IIT Kharagpur

Participants must recover AES keys using side-channel analysis on two datasets: simulated and real power traces.  
> Problem 1 uses clean, simulated traces with ciphertext and power samples.

> Problem 2 uses noisy, real hardware traces with plaintext, ciphertext, and power data.  
    1. Each team submits the full 16-byte key and ranked guesses for each byte (256 values per byte).  
    2. Scoring is split: 50% for exact key recovery, 50% for ranking accuracy using metrics like reciprocal rank.  
    3. Real traces require denoising and alignment techniques for better results.  
    4. The assumed leakage model is Hamming Weight of intermediate values.  
    5. Any statistical, analytical, or ML methods are allowed.  
    6. Final rankings are based on combined scores across both problems.
