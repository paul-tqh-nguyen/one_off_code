
# For the second checkout 
# second checkout translation
# cd ~/build/CycJava-polaris/Linux-maaz/second && ./run-ant.sh clean && cd ~/build/linux-maaz/allegro-altair/second && ./update-classpath.sh && svn update ~/src/second/cycorp/cyc && rm -rf ~/src/second/cycorp/cyc/java/tool/subl/src/com/cyc/cycjava/cycl/* && cd ~/build/cmucl/second && cmucl -init ~/build/cmucl/second/setup/translation-manual-java.lisp
# second checkout jmake
# cd ~/build/CycJava-polaris/Linux-maaz/second && rm -rf ./bin && rm -rf ./lib && ./run-ant.sh clean && ./run-ant.sh

# Set the checkout
[ -z "$SRC_TAG" ] && export SRC_TAG=head
alias use-head="export SRC_TAG=head && source ~/.bashrc.local"
alias use-second="export SRC_TAG=second && source ~/.bashrc.local"
alias use-third="export SRC_TAG=third && source ~/.bashrc.local"
alias use-fourth="export SRC_TAG=fourth && source ~/.bashrc.local"

# Cyc-specific
alias cd-jrtl-build-dir="cd ~/build/CycJava-polaris/Linux-maaz/$SRC_TAG"
function cyc() {
  cd-jrtl-build-dir
  echo "Running Cyc out of the directory " $(pwd)
  echo "Using the units directory " $UNITS_DIR
  /cyc/top/scripts/cyc-dev.sh "$@"
}
alias update-jlinker-class-path="cd ~/build/linux-maaz/allegro-altair/$SRC_TAG && ./update-classpath.sh"
alias jmake="cd-jrtl-build-dir && rm -rf ./bin && rm -rf ./lib && ./run-ant.sh clean && ./run-ant.sh"
alias testt="jmake && cd-jrtl-build-dir && cyc tiny test enable-terse-logging"
alias testtnow="cd-jrtl-build-dir && cyc tiny test enable-terse-logging"
alias testfnow="cd-jrtl-build-dir && cyc test core no-tiny-in-full"
alias testfsoon="sleep 30; testfnow"
alias testf="sleep 180; testfnow"
alias jtrans="echo 'JTRANS start time: ' $(date) $'\n' && cd-jrtl-build-dir && ./run-ant.sh clean && update-jlinker-class-path && svn update ~/src/$SRC_TAG/cycorp/cyc && rm -rf ~/src/$SRC_TAG/cycorp/cyc/java/tool/subl/src/com/cyc/cycjava/cycl/* && cd ~/build/cmucl/$SRC_TAG && cmucl -init ~/build/cmucl/$SRC_TAG/setup/translation-manual-java.lisp"
alias time-jtrans="echo 'JTRANS start time: ' $(date) $'\n' && update-jlinker-class-path && svn update ~/src/$SRC_TAG/cycorp/cyc && rm -rf ~/src/$SRC_TAG/cycorp/cyc/java/tool/subl/src/com/cyc/cycjava/cycl/* && cd ~/build/cmucl/$SRC_TAG && cmucl -init ~/build/cmucl/$SRC_TAG/setup/translation-manual-java.lisp -eval '(cyc::show-build-problems)(quit)' && echo 'JTRANS end time: ' $(date) $'\n'"
alias cd-crtl-build-dir="cd /home/pnguyen/build/Linux-maaz/$SRC_TAG"
alias ctrans="svn update ~/src/$SRC_TAG/cycorp/cyc && cd ~/build/cmucl/$SRC_TAG && cmucl -init ~/build/cmucl/$SRC_TAG/setup/translation-manual-c.lisp"
alias crtl-make="cd-crtl-build-dir && make"
alias crtl-make-full-kb="cd-crtl-build-dir && make world"
alias crtl-run-tiny-tests="cd-crtl-build-dir && make tiny-test"
alias crtl-run-full-tests="cd-crtl-build-dir && make test"

#these need to be fixed
#starting crtl template
# ./cyc -w ./world/kb-1356.load -i ./init/offset-services-init-00.lisp
#alias crtl-cyc="cd-crtl-build-dir && ./"
#alias crtl-build-tiny-world="./cyc -b -q -i init/cyc-init.lisp -f '(progn (load "'/cyc/images/snapshot-units/1366/1366op8864/core-kb.lisp'") (build-write-image "world/'1366op8864-tiny.load'"))'" 
#alias crtl-test="./cyc -b -q -w world/1366op8864-tiny.load -i init/cyc-init.lisp"

# Convenience & Workflow 
alias svndiff1="svn diff --diff-cmd=diff -x -U1"
alias svndiff10="svn diff --diff-cmd=diff -x -U10"
alias svndiff20="svn diff --diff-cmd=diff -x -U20"
alias svndiff30="svn diff --diff-cmd=diff -x -U30"
alias svndiff40="svn diff --diff-cmd=diff -x -U40"
alias svndiff50="svn diff --diff-cmd=diff -x -U50"
alias svndiff60="svn diff --diff-cmd=diff -x -U60"
alias svndiff70="svn diff --diff-cmd=diff -x -U70"
alias svndiff80="svn diff --diff-cmd=diff -x -U80"
alias svndiff90="svn diff --diff-cmd=diff -x -U90"
alias cd-cycl="cd ~/src/$SRC_TAG/cycorp/cyc/cyc-lisp/cycl/"
alias lt="ls -ltr"
alias myjobs="ps auwwx | grep $USER"
alias open="xdg-open"

# Other
export PATH=$PATH:~/scripts/:~/bin/

function set-title() {
  if [[ -z "$ORIG" ]]; then
    ORIG=$PS1
  fi
  TITLE="\[\e]2;$*\a\]"
  PS1=${ORIG}${TITLE}
}