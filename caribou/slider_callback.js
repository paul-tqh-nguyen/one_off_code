
const dateFromInt = (value) => new Date(value + (new Date(value).getTimezoneOffset() * 60 * 1000));
const dateStringToInt = (dateString) => new Date(dateString).getTime();
const dateFromString = (dateString) => dateFromInt(dateStringToInt(dateString));
const dateToString = (date) => `${date.getFullYear()}-${((date.getMonth() > 8) ? (date.getMonth() + 1) : ('0' + (date.getMonth() + 1)))}-${((date.getDate() > 9) ? date.getDate() : ('0' + date.getDate()))}`;
const dayBeforeDate = (date) => {
    const yesterday = new Date(date.getTime());
    yesterday.setDate(date.getDate() - 1);
    return yesterday;
};
const dayAfterDate = (date) => {
    const tomorrow = new Date(date.getTime());
    tomorrow.setDate(date.getDate() + 1);
    return tomorrow;
};
const lerp = (start, end, interpolationAmount) => start + interpolationAmount * (end - start);

const generateAnimalIdToEarlierDateString = (date, startDate, remainingAnimalIds, locationByDateString) => {
    const animalIdToEarlierDateString = {};
    const remainingAnimalIdsWithoutEarlierDate = new Set(remainingAnimalIds);
    let currentDate = date;
    while (startDate < currentDate && remainingAnimalIdsWithoutEarlierDate.size > 0) {
        currentDate = dayBeforeDate(currentDate);
        const currentDateString = dateToString(currentDate);
        const locations = locationByDateString[currentDateString];
        const animalIdsFound = new Set();
        if (locations) {
            Object.keys(locations).forEach((animalId) => {
                if (remainingAnimalIds.has(animalId)) {
                    animalIdToEarlierDateString[animalId] = currentDateString;
                    animalIdsFound.add(animalId);
                }
            });
        }
        animalIdsFound.forEach((animalId) => remainingAnimalIdsWithoutEarlierDate.delete(animalId));
    }
    return animalIdToEarlierDateString;
};  

const generateAnimalIdToLaterDateString = (date, endDate, remainingAnimalIds, locationByDateString) => {
    const animalIdToLaterDateString = {};
    const remainingAnimalIdsWithoutLaterDate = new Set(remainingAnimalIds);
    let currentDate = date;
    while (currentDate < endDate && remainingAnimalIdsWithoutLaterDate.size > 0) {
        currentDate = dayAfterDate(currentDate);
        const currentDateString = dateToString(currentDate);
        const locations = locationByDateString[currentDateString];
        const animalIdsFound = new Set();
        if (locations) {
            Object.keys(locations).forEach((animalId) => {
                if (remainingAnimalIds.has(animalId)) {
                    animalIdToLaterDateString[animalId] = currentDateString;
                    animalIdsFound.add(animalId);
                }
            });
        }
        animalIdsFound.forEach((animalId) => remainingAnimalIdsWithoutLaterDate.delete(animalId));
    }
    return animalIdToLaterDateString;
};

const main = (dateSlider, caribouCirclesDataSource, locationByDateString, animalIds, startDateString, endDateString) => {
    console.clear();
    const date = dateFromInt(dateSlider.value);
    const startDate = dateFromString(startDateString);
    const endDate = dateFromString(endDateString);
    
    const dateString = dateToString(date);
    const locationsForDateSliderValue = locationByDateString[dateString];
    const animalIdToLocation = locationsForDateSliderValue ? locationsForDateSliderValue : {};
    
    if (!locationsForDateSliderValue || Object.keys(locationsForDateSliderValue).length !== animalIds.length) {
        const remainingAnimalIds = new Set(animalIds);
        if (locationsForDateSliderValue) {
            Object.keys(locationsForDateSliderValue).forEach((animalId) => remainingAnimalIds.delete(animalId));
        }
        const animalIdToEarlierDateString = generateAnimalIdToEarlierDateString(date, startDate, remainingAnimalIds, locationByDateString);
        const animalIdToLaterDateString = generateAnimalIdToLaterDateString(date, endDate, remainingAnimalIds, locationByDateString);
        animalIds.forEach((animalId) => {
            const earlierDateString = animalIdToEarlierDateString[animalId];
            const laterDateString = animalIdToLaterDateString[animalId];
            if (earlierDateString && laterDateString) {
                const earlierLocation = locationByDateString[earlierDateString][animalId];
                const laterLocation = locationByDateString[laterDateString][animalId];
                const interpolationAmount = (dateSlider.value - dateStringToInt(earlierDateString)) / (dateStringToInt(laterDateString) - dateStringToInt(earlierDateString));
                animalIdToLocation[animalId] = {
                    'longitude_x': lerp(earlierLocation.longitude_x, laterLocation.longitude_x, interpolationAmount),
                    'latitude_y': lerp(earlierLocation.latitude_y, laterLocation.latitude_y, interpolationAmount),
                };
            }
        });
    }
    if (Object.keys(animalIdToLocation).length > 0) {
        const caribouCount = caribouCirclesDataSource.data.index.length;
        for (let i = 0; i < caribouCount; i++) {
            const animalId = caribouCirclesDataSource.data.animal_id[i];
            const caribouLocation = animalIdToLocation[animalId];
            if (caribouLocation) {
                caribouCirclesDataSource.data.longitude_x[i] = caribouLocation.longitude_x;
                caribouCirclesDataSource.data.latitude_y[i] = caribouLocation.latitude_y;
                caribouCirclesDataSource.data.alpha[i] = 1.0;
            } else {
                caribouCirclesDataSource.data.alpha[i] = 0.0;
            }
        }
        caribouCirclesDataSource.change.emit();
    }
};

main(dateSlider, caribouCirclesDataSource, locationByDateString, animalIds, startDateString, endDateString);
