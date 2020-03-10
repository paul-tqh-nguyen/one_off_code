
# Convenience & Workflow 
alias lt="ls -ltr"
alias myjobs="ps auwwx | grep $USER"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    alias open="xdg-open"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    alias ll="ls -l"
else
    echo "Could not detect OS flavor."
fi

# Other
export PATH=$PATH:~/scripts/:~/bin/

function set-title() {
  if [[ -z "$ORIG" ]]; then
    ORIG=$PS1
  fi
  TITLE="\[\e]2;$*\a\]"
  PS1=${ORIG}${TITLE}
}
