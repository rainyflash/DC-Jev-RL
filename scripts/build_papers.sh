#!/bin/bash
# build_papers.sh <papers dir> [placeholders]: compile every main_en.tex (pdflatex) and main_zh.tex (xelatex) with
# latexmk. With "placeholders", a missing \input file is created empty first (a trial build of unfinished papers).
# Prints, per paper, the exit code, undefined numbers/references/citations and the first LaTeX errors.
cd "$1" || exit 1
for d in part1_control part2_grpo part3a_code part3b_labels part3c_tree; do
  [ -d "$d" ] || continue
  cd "$d"
  for m in main_en main_zh; do
    [ -f $m.tex ] || continue
    if [ "$2" = placeholders ]; then
      for f in $(grep -ho '\\input{[^}]*}' $m.tex | sed 's/^\\input{\(.*\)}$/\1/' | grep -v '^\.\./'); do
        [ -f "$f.tex" ] || [ -f "$f" ] || { echo "% placeholder for a trial build" > "$f.tex"; echo "  ($d: placeholder $f.tex)"; }
      done
    fi
    if [ $m = main_en ]; then eng=-pdf; else eng=-xelatex; fi
    latexmk $eng -interaction=nonstopmode -halt-on-error -f $m.tex > $m.build.log 2>&1
    rc=$?
    pages=$(grep -ao 'Output written on [^ ]* ([0-9]* page' $m.log 2>/dev/null | tail -1 | grep -o '([0-9]*' | tr -d '(')
    echo "$d/$m: exit $rc, ${pages:-?} pages"
    grep -a 'Undefined number' $m.log | sed 's/.*Undefined number \([^ ]*\).*/  undefined number: \1/' | sort -u | head -40
    grep -a "Reference .* undefined\|Citation .* undefined" $m.log | sed 's/^LaTeX Warning: //' | sort -u | head -20
    grep -a -A3 '^! ' $m.log | head -20
    grep -a -A1 'Overfull \\hbox' $m.log | grep -v '^--$' | head -16
  done
  cd ..
done
