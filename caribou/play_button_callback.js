
const onlyOne = (iterable) => {
    console.assert(iterable.length == 1);
    return iterable[0];
};

const main = () => {
    const playButtonDiv = onlyOne(document.getElementsByClassName('play-pause-button'));
    const playButton = onlyOne(playButtonDiv.getElementsByTagName('button'));
    playButton.innerHTML = playButton.innerHTML === 'Pause' ? 'Play' : 'Pause';

    const timer = timerBox[0];
};

main();
