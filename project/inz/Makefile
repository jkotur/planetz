NAME=inż

all: pdf

clean:
	rm $(NAME).aux $(NAME).log $(NAME).toc $(NAME).out $(NAME).bbl $(NAME).blg

pdf:
	pdflatex $(NAME).tex

bib:
	pdflatex $(NAME).tex
	bibtex $(NAME).aux
	pdflatex $(NAME).tex
	pdflatex $(NAME).tex


run: pdf
	evince $(NAME).pdf

bibrun: bib run

