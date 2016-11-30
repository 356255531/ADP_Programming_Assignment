make: Makefile main.py
	rm -f Plots/*y
	python main.py
	rm -f Toolbox/*.pyc
