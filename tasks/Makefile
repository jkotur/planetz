NAME=gantt

all: pdf

clean:
	rm $(NAME).aux $(NAME).log $(NAME).toc $(NAME).out

pdf:
	pdflatex $(NAME).tex

run: pdf
	evince $(NAME).pdf
