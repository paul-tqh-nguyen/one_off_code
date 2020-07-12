
const implies = (antecedent, consequent) => (!antecedent) || consequent;
const onlyOne = (iterable) => {
    console.assert(iterable.length == 1);
    return iterable[0];
};

const timerPeriod = 25;

const main = (dateSlider, timerBox) => {
    const playButtonDiv = onlyOne(document.getElementsByClassName('play-pause-button'));
    const playButton = onlyOne(playButtonDiv.getElementsByTagName('button'));
    const originalPlayButtonText = playButton.innerHTML;
    
    const timer = onlyOne(timerBox);
    if (timer !== null) {
        clearInterval(timer);
    }
    if (originalPlayButtonText  === 'Play') {
        if (dateSlider.value >= dateSlider.end) {
            dateSlider.value = dateSlider.start;
        }
        timerBox[0] = setInterval(() => {
            if (dateSlider.value >= dateSlider.end) {
                playButton.innerHTML = 'Play';
                clearInterval(timerBox[0]);
            } else {
                dateSlider.value += 1000 * 3600 * 24;
            }
        }, timerPeriod);
    }
    playButton.innerHTML = originalPlayButtonText === 'Pause' ? 'Play' : 'Pause';
};

main(dateSlider, timerBox);
