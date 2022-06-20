default: gan

gan:
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 python3 GAN.py
