% This template aims to simplify and improve the (Xe)LaTeX template provided
% by the TU Delft. Original template by TU Delft. Rewritten template by Daan
% Zwaneveld (https://dzwaneveld.github.io)
%
% This template by Daan Zwaneveld is licensed under CC BY-NC 4.0. To view a
% copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/.
% No attribution is required in PDF outputs created using this template.

%% ----------------------------------------------------------------------
%%    Setting up the class, main packages and basic definitions
%% ----------------------------------------------------------------------

%% Class is based on the default book class and options will be passed
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tudelft-report}[05-09-2024 v1.7.1 TU Delft Report Class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass[10pt,oneside]{book}

%% Main packages in the document --- Some are imported later in the class file
\RequirePackage{mathtools}  % Mathematical tools to use with amsmath
\RequirePackage{amssymb}    % Extended symbol collection
\RequirePackage{siunitx}    % Comprehensive (SI) units package

\RequirePackage{tabularx}   % Tabulars with adjustable-width columns
\RequirePackage{booktabs}   % Publication quality tables
\RequirePackage{longtable}  % Allow tables to flow over page boundaries
\RequirePackage{multirow}   % Create tabular cells spanning multiple rows

\RequirePackage{graphicx}   % Enhanced support for images
\RequirePackage{float}      % Improved interface for floating objects
\RequirePackage[labelfont=bf,justification=centering,footnotesize]{caption} % Captions
\RequirePackage{subcaption} % Support for sub-captions
\RequirePackage{pdfpages}   % Include PDF documents

\RequirePackage[pdfusetitle,hidelinks]{hyperref} % Extensive support for hypertext
\RequirePackage[noabbrev]{cleveref} % Intelligent cross-referencing
\RequirePackage{xcolor}     % Driver-independent color extensions
\RequirePackage{tikz}       % Create PostScript and PDF graphics
\RequirePackage{xspace}     % Define commands that appear not to eat spaces
\RequirePackage{microtype}  % Refinements towards typographical perfection

\RequirePackage{geometry}   % Customize document dimensions
\RequirePackage{titlesec}   % Select alternative section titles
\RequirePackage{titletoc}   % Alternative headings for toc
\RequirePackage{fancyhdr}   % Control of page headers and footers
\RequirePackage{enumitem}   % Control layout of itemize, enumerate, description
\RequirePackage{etoolbox}   % Toolbox of programming facilities
\RequirePackage{iftex}      % Adds if-else statements to support multiple compilers
\RequirePackage{datetime}   % Change format of \today

%% Establish commands for the subtitle, subject, affiliation, cover image and table of authors
\newcommand*{\subtitle}[1]{\def\@subtitle{#1}}
\newcommand*{\subject}[1]{\def\@subject{#1}}
\newcommand*{\affiliation}[1]{\def\@affiliation{#1}}
\newcommand*{\coverimage}[1]{\def\@coverimage{#1}}
\newcommand*{\covertable}[1]{\def\@covertable{#1}}

%% Scale the margins to be slightly smaller than default (.7)
\geometry{a4paper,hscale=0.75,vscale=0.8}

%% ----------------------------------------------------------------------
%%    Setting up the fonts
%% ----------------------------------------------------------------------

\ifPDFTeX
    %% With pdfLaTeX, use Paletino as the main font and Roboto Slab as title fonts
    \RequirePackage[T1]{fontenc}  % Select T1 font encoding
    \RequirePackage{newpxtext}    % Palatino-like font...
    \RequirePackage{newpxmath}    % ...with support for mathematics

    \newcommand{\titlestyle}{\fontfamily{Roboto-LF}\fontseries{light}\selectfont}
    \newcommand{\largetitlestyle}{\fontfamily{Roboto-LF}\fontseries{thin}\selectfont}

    %% Cover and title page: Roboto (sans) Medium for the title, Roboto Light for subtitle and author
    \newcommand{\covertitlestyle}{\fontfamily{Roboto-LF}\fontseries{medium}\selectfont}
    \newcommand{\coversubtitlestyle}{\fontfamily{Roboto-LF}\fontseries{light}\selectfont}
    \newcommand{\coverregularstyle}{\fontfamily{Roboto-LF}\fontseries{regular}\selectfont}
\else
    %% If XeLaTeX or LuaLaTeX is set as the compiler, the TU Delft house style fonts are used (see https://www.tudelft.nl/huisstijl/bouwstenen/typografie).
    \RequirePackage[no-math]{fontspec} % Advanced font selection

    %% Use Arial as the main font and Roboto Slab as title fonts
    \setmainfont{Arial}
    \setmathsf{Arial}
    \setmathtt{Arial}

    \newfontfamily\titlestyle{Roboto Light}
    \newfontfamily\largetitlestyle{Roboto Thin}
    \newfontfamily\covertitlestyle{Roboto Medium}
    \newfontfamily\coversubtitlestyle{Roboto Light}
    \newfontfamily\coverregularstyle{Roboto}
\fi

%% ----------------------------------------------------------------------
%%    Formatting the titles and table of contents
%% ----------------------------------------------------------------------

%% Format the chapter titles and spacing
\titleformat{\chapter}[display]
    {\flushright}
    {\fontsize{96}{96}\selectfont\largetitlestyle\thechapter}
    {0pt}
    {\Huge\titlestyle}
\titlespacing*{\chapter}{0pt}{0pt}{2\baselineskip}

%% Format the section titles and spacing
\titleformat{\section}
    {\Large\titlestyle\bfseries}
    {\thesection.}
    {5pt}
    {}
\titlespacing*{\section}{0pt}{\baselineskip}{0pt}

%% Format the subsections titles and spacing
\titleformat{\subsection}
    {\large\titlestyle\bfseries}
    {\thesubsection.}
    {5pt}
    {}
\titlespacing*{\subsection}{0pt}{\baselineskip}{0pt}

%% Format the subsubsections titles and spacing
\titleformat{\subsubsection}
    {\titlestyle\bfseries}
    {\thesubsubsection.}
    {5pt}
    {}
\titlespacing*{\subsubsection}{0pt}{\bigskipamount}{0pt}

%% Reduce the vertical white space between chapters in the table of contents
\dottedcontents{chapter}[1.5em]{\vspace{0.5\baselineskip}\bfseries}{1.5em}{0pt}

%% ----------------------------------------------------------------------
%%    Formatting the header and footer
%% ----------------------------------------------------------------------

%% Format the header and footer of 'plain' pages
\fancypagestyle{plain}{%
    \fancyhf{}
    \renewcommand{\headrulewidth}{0pt}
    \fancyfoot[C]{\titlestyle\thepage}}

%% Format the header and footer of 'fancy' pages (based on twoside option)
\if@twoside
    \fancyhf{}
    \fancyhead[LE,RO]{\titlestyle\thepage}
    \fancyhead[RE]{\titlestyle\nouppercase{\leftmark}}
    \fancyhead[LO]{\titlestyle\nouppercase{\rightmark}}
    \RequirePackage{emptypage} % Remove header and footer on empty pages
\else
    \fancyhf{}
    \fancyhead[R]{\titlestyle\thepage}
    \fancyhead[L]{\titlestyle\nouppercase{\rightmark}}
\fi

\pagestyle{fancy} % Set the package defaults and the additional changes as the style

%% ----------------------------------------------------------------------
%%    Setting up the \makecover command for the cover page
%% ----------------------------------------------------------------------

\newcommand*{\makecover}{
    %% Use the Tikz library positioning and clear the page header and footer
    \usetikzlibrary{positioning}
    \thispagestyle{empty}

    %% Construct the cover page with Tikz
    \begin{tikzpicture}[overlay,remember picture]

        %% Add the cover image
        \node[above=0,inner sep=0] at (current page.south) {%
            \includegraphics[width=\paperwidth]{\@coverimage}};

        %% Add the affiliation on the left
        %% (dark green sampled from the foliage of the cover photo)
        \definecolor{affiliation}{HTML}{1B211A}
        \node[rotate=90,below right=15mm and 3mm] at (current page.west) {%
            \ifdefvoid{\@affiliation}{}{\color{affiliation}\titlestyle\@affiliation}};

        %% Add the banner with the TU Delft, PoliTo, AOU and FEV logos at the bottom
        \node[above=0,fill=black!85,minimum width={\paperwidth},inner ysep=10mm,opacity=0.6,text opacity=1] at (current page.south) {%
            \makebox[0.9\paperwidth]{%
                \raisebox{-0.5\height}{\includegraphics[height=15mm]{figures/cover_and_logos/logo-white.pdf}}\hfill
                \raisebox{-0.5\height}{\includegraphics[height=17mm]{figures/cover_and_logos/PoliTO_CMYK_bianco-logo.png}}\hfill
                \raisebox{-0.5\height}{\includegraphics[height=19mm]{figures/cover_and_logos/logo_AOU_white.png}}\hfill
                \raisebox{-0.5\height}{\includegraphics[height=11mm]{figures/cover_and_logos/FEV_logo_white.png}}%
            }};

        %% Add the banner with the title, subtitle, subject and author(s)
        \node[below=2cm,fill=black!85,minimum width={\paperwidth},inner ysep=25pt,opacity=0.8,text opacity=1] at (current page.north) {%
            \begin{minipage}{0.9\paperwidth}\begin{minipage}{0.75\paperwidth}
                %% Format and add the title (narrower box so the title breaks after "System")
                \color{white}\raggedright\covertitlestyle\fontsize{30}{34.5}\selectfont%
                    \@title\par\vspace{0.8ex}
                %% Format and add (optional) subtitle and subject
                \coverregularstyle\fontsize{24}{28.8}\selectfont%
                    \ifdefvoid{\@subtitle}{}{\@subtitle\par\vspace{3ex}}%
                    \ifdefvoid{\@subject}{}{\@subject\par\vspace{0.5ex}}%
                %% Format and add author or table of authors
                \coverregularstyle\fontsize{18}{21.6}\selectfont%
                    \ifdefvoid{\@covertable}{\@author}{\@covertable}
            \end{minipage}\hfill\end{minipage}};

    \end{tikzpicture}
    \newpage
}

%% ----------------------------------------------------------------------
%%    Setting up listings
%% ----------------------------------------------------------------------

\RequirePackage{listings} % Typeset source code listings

\lstset{
  basicstyle=\ttfamily\footnotesize,  % Style of the font that is used for the code
  backgroundcolor=\color{gray!10},    % Background color
  keywordstyle=\color{red!75!black},  % Keyword style
  stringstyle=\color{green!40!black}, % String style
  commentstyle=\color{blue!30!black}, % Comment style
  numbers=left,                       % Add line numbers on the left side
  numbersep=5pt,                      % Decrease distance between line numbers and code
  numberstyle=\tiny,                  % Line number style
  breaklines=true,                    % Line break automatically
}
