NAME=inż
TITLE=strona_tytułowa
INTRO=streszczenie

all: pdf

clean:
	for x in $(NAME) $(TITLE) $(INTRO) ;\
	do \
		rm $$x.aux $$x.log $$x.toc $$x.out $$x.bbl $$x.blg 2> /dev/null ;\
	done 


pdf:
	pdflatex $(NAME).tex

title:
	pdflatex $(TITLE).tex

intro:
	pdflatex $(INTRO).tex

bib: pdf
	bibtex $(NAME).aux
	pdflatex $(NAME).tex
	pdflatex $(NAME).tex


run: pdf
	evince $(NAME).pdf

bibrun: bib run

