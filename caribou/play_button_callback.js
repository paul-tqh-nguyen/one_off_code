
const implies = (antecedent, consequent) => (!antecedent) || consequent;
const onlyOne = (iterable) => {
    console.assert(iterable.length == 1);
    return iterable[0];
};

const timerPeriod = 1;

const main = (dateSlider, timerBox) => {
    const playButtonDiv = onlyOne(document.getElementsByClassName('play-pause-button'));
    const playButton = onlyOne(playButtonDiv.getElementsByTagName('button'));
    
    const timer = onlyOne(timerBox);
    if (timer !== null) {
        clearInterval(timer);
    }
    if (playButton.innerHTML  === 'Play') {
        if (dateSlider.value >= dateSlider.end) {
            dateSlider.value = dateSlider.start;
        }
        timerBox[0] = setInterval(() => {
            if (dateSlider.value >= dateSlider.end) {
                clearInterval(timerBox[0]);
            } else {
                dateSlider.value += 1000 * 3600 * 24;
            }
        }, timerPeriod);
    }
    playButton.innerHTML = playButton.innerHTML === 'Pause' ? 'Play' : 'Pause';
};

main(dateSlider, timerBox);
