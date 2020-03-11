
# Convenience & Workflow 

alias lt="ls -ltr"
alias myjobs="ps auwwx | grep $USER"
export PS1="\u@\h:\`pwd\`$ "

function filesize {
    num_bytes=$(cat $1 | wc --bytes)
    if [ "1024" -gt "$num_bytes" ]
    then
	echo $(echo "$num_bytes") bytes
    else
	echo $(echo $num_bytes | numfmt --to=iec-i)
    fi
}  

function dirsize {
    echo $(du -h $1 | tail -n 1 | awk '{print $1}')
}

function set-title() {
  if [[ -z "$ORIG" ]]; then
    ORIG=$PS1
  fi
  TITLE="\[\e]2;$*\a\]"
  PS1=${ORIG}${TITLE}
}

# OS Specific Basic Needs

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    bind 'set bell-style none'
    alias open="xdg-open"
    export PATH=/usr/local/cuda/bin${PATH:+:$PATH}
    export PATH=/home/pnguyen/.local/bin${PATH:+:$PATH}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export PATH=$PATH:~/scripts/:~/bin/
elif [[ "$OSTYPE" == "darwin"* ]]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    export PATH=/Users/pnguyen/anaconda3/condabin${PATH:+:$PATH}
    export PATH=/opt/anaconda3/bin${PATH:+:$PATH}
    alias ll="ls -alF"
else
    echo "Could not detect OS flavor."
fi
