
const implies = (antecedent, consequent) => (!antecedent) || consequent;
const onlyOne = (iterable) => {
    console.assert(iterable.length == 1);
    return iterable[0];
};

const timerPeriod = 25;

const main = () => {
    const playButtonDiv = onlyOne(document.getElementsByClassName('play-pause-button'));
    const playButton = onlyOne(playButtonDiv.getElementsByTagName('button'));
    
    const dateSliderDiv = onlyOne(document.getElementsByClassName('date-slider'));
    const dateSliderHandleDiv = onlyOne(dateSliderDiv.querySelectorAll('div.noUi-handle.noUi-handle-lower'));
    
    const timer = onlyOne(timerBox);
    if (timer !== null) {
        clearInterval(timer);
    }
    if (playButton.innerHTML  === 'Play') {
        timerBox[0] = setInterval(() => {
            const sliderValue = parseInt(dateSliderHandleDiv.getAttribute('aria-valuenow'));
            const sliderNextDayValueString = `${sliderValue + 1000 * 3600 * 24}.0`;
            dateSliderHandleDiv.setAttribute('aria-valuenow', sliderNextDayValueString);
        }, timerPeriod);
    }
    playButton.innerHTML = playButton.innerHTML === 'Pause' ? 'Play' : 'Pause';
};

main();
