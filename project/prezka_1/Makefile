FILE=inż_1
all: pdf

clean:
	rm $(FILE).aux
	rm $(FILE).log
	rm $(FILE).nav
	rm $(FILE).out
	rm $(FILE).snm
	rm $(FILE).toc

pdf:
	pdflatex $(FILE).tex

run: pdf
	evince $(FILE).pdf
