make: Makefile main.py clean
	rm -f Plots/*y *.jpg
	python main.py
	rm -f Toolbox/*.pyc

clean:
	rm -f *.jpg
