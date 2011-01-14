FILE=inż_2
all: pdf

clean:
	rm $(FILE).aux
	rm $(FILE).log
	rm $(FILE).nav
	rm $(FILE).out
	rm $(FILE).snm
	rm $(FILE).toc

pdf:
	TEXINPUTS='.:../include/:' pdflatex $(FILE).tex

run: pdf
	evince $(FILE).pdf
