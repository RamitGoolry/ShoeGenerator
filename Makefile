default: gan

gan:
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 python3 GAN.py
