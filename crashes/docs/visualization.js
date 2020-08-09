
const sum = inputArray => inputArray.reduce((a, b) => a + b, 0);
const mean = inputArray => sum(inputArray) / inputArray.length;

d3.selection.prototype.moveToFront = function() {
    return this.each(function() {
        if (this.parentNode !== null) {
            this.parentNode.appendChild(this);
        }
    });
};

let updateDate;

const visualizationMain = () => {
    
    const boroughJSONLocation = './processed_data/borough.json';
    const zipCodesJSONLocation = './processed_data/zip_code.json';
    const crashDateFilesJSONLocation = './processed_data/crash_date_files.json';

    const svg = d3.select('#map-svg');

    const zoomableContentGroup = svg.append('g').attr('id', 'zoomable-content-group');
    const boroughsGroup = zoomableContentGroup.append('g').attr('id', 'boroughs-group');
    const zipCodesGroup = zoomableContentGroup.append('g').attr('id', 'zip-codes-group');
    const crashesGroup = zoomableContentGroup.append('g').attr('id', 'crashes-group');

    const dateDropdown = document.getElementById('date-dropdown');

    const zoomLevels = {
        CITY: 'city',
        BOROUGH: 'borough',
        ZIPCODE: 'zip_code',
    };
    let zoomLevel = zoomLevels.CITY;
    const zoomTransitionTime = 500;
    
    const boroughToHexColor = {
        'Bronx': '#8000ff',
        'Staten Island': '#00b5ec',
        'Manhattan': '#81ffb4',
        'Brooklyn': '#ffb360',
        'Queens': '#de2121'
    };
    
    Promise.all([
        d3.json(boroughJSONLocation),
        d3.json(zipCodesJSONLocation),
        d3.json(crashDateFilesJSONLocation),
    ]).then(data => {
        
        const [boroughData, zipCodeData, crashDateFileData]  = data;

        const sortedDateStrings = Object.keys(crashDateFileData).sort((dateA, dateB) =>  new Date(dateB.date) - new Date(dateA.date));
        sortedDateStrings.forEach(dateString => {
            const optionElement = document.createElement('option');
            optionElement.setAttribute('value', dateString);
            optionElement.innerHTML = dateString.split("T")[0];
            dateDropdown.append(optionElement);
        });
        const crashDateDataByDateString = sortedDateStrings.reduce((accumulator, dateString) => {
            accumulator[dateString] = null;
            return accumulator;
	}, {});
        let updateVisualizationWithDateData;
        updateDate = () => {
            const dateString = dateDropdown.value;
            if (crashDateDataByDateString[dateString] !== null) {
                updateVisualizationWithDateData();
            } else {
                const crashDateFile = crashDateFileData[dateString];
                d3.json(crashDateFile)
                    .then(data => {
                        crashDateDataByDateString[dateString] = data[dateString];
                        updateVisualizationWithDateData();
                    }).catch((error) => {
                        console.error(`Could not load data at ${crashDateFile} for date ${dateString}`);
                        console.error(error);
                    });
            }
        };
        const boroughToMeanLongLat = boroughData.features.reduce((accumulator, feature) => {
            const [meanLong, meanLat] = feature.geometry.coordinates.map(polygon => polygon[0])
                  .reduce((a, b) => a.concat(b),[])
                  .reduce((longLatAccumulator, longLat) => {
                      longLatAccumulator[0] += longLat[0];
                      longLatAccumulator[1] += longLat[1];
                      return longLatAccumulator;
                  }, [0,0]);
            accumulator[feature.properties.boro_name] = {meanLong: meanLong, meanLat: meanLat};
            return accumulator;
        }, {});
        
        const render = () => {
            
            boroughsGroup.selectAll('*').remove();
            zipCodesGroup.selectAll('*').remove();
            crashesGroup.selectAll('*').remove();
            
            const plotContainer = document.getElementById('visualization-display');
            svg
	        .attr('width', 0)
	        .attr('height', 0)
	        .attr('width', `${plotContainer.clientWidth}px`)
	        .attr('height', `${plotContainer.clientHeight}px`);

            const svgWidth = parseFloat(svg.attr('width'));
	    const svgHeight = parseFloat(svg.attr('height'));

            const projection = d3.geoMercator().fitExtent([[0, 0], [svgWidth, svgHeight]], boroughData);
            const projectionFunction = d3.geoPath().projection(projection);

            const zoom = d3.zoom()
                  .scaleExtent([1, 8])
                  .on("zoom", () => zoomableContentGroup.attr("transform", d3.event.transform));

            let zipCodesGroupTransitionOutTimer = null;
            const moveBoroughGroupToFront = () => {
                zipCodesGroup
                    .transition()
                    .duration(zoomTransitionTime)
                    .attr('stroke-opacity', 0);
                if (zipCodesGroupTransitionOutTimer !== null) {
                    clearTimeout(zipCodesGroupTransitionOutTimer);
                }
                zipCodesGroupTransitionOutTimer = setTimeout(() => {
                    if (zoomLevel === zoomLevels.CITY) {
                        boroughsGroup.moveToFront();
                    }
                }, zoomTransitionTime*4);
            };
            const moveZipCodesGroupToFront = () => {
                zipCodesGroup
                    .transition()
                    .duration(zoomTransitionTime)
                    .attr('stroke-opacity', 1);
                zipCodesGroup.moveToFront();
            };

            const zoomToPath = datum => {
                const X_COORD = 0;
                const Y_COORD = 1;
                const bounds = projectionFunction.bounds(datum);
                const dx = bounds[1][X_COORD] - bounds[0][X_COORD];
                const dy = bounds[1][Y_COORD] - bounds[0][Y_COORD];
                const x = (bounds[0][X_COORD] + bounds[1][X_COORD]) / 2;
                const y = (bounds[0][Y_COORD] + bounds[1][Y_COORD]) / 2;
                const scale = Math.max(1, Math.min(8, 0.9 / Math.max(dx / svgWidth, dy / svgHeight)));
                const translate = [svgWidth / 2 - scale * x, svgHeight / 2 - scale * y];
                zoomableContentGroup.transition()
                    .duration(zoomTransitionTime)
                    .call( zoom.transform, d3.zoomIdentity.translate(translate[0],translate[1]).scale(scale) );
            };
            const zoomReset = () => {
                zoomableContentGroup.transition()
                    .duration(zoomTransitionTime)
                    .call(zoom.transform, d3.zoomIdentity);
                zoomLevel = zoomLevels.CITY;
                moveBoroughGroupToFront();
            };
            boroughsGroup.moveToFront();
            zoomReset();
            
            zipCodesGroup
                .selectAll('path')
                .data(zipCodeData.features)
                .enter()
                .append('path')
                .attr('class', 'zip-code-path')
                .attr('d', datum => projectionFunction(datum))
                .on("click", function (datum) {
                    if (zoomLevel === zoomLevels.ZIPCODE) {
                        zoomReset();
                    } else if (zoomLevel === zoomLevels.BOROUGH) {
                        zoomToPath(datum);
                        zoomLevel = zoomLevels.ZIPCODE;
                    } else if (zoomLevel === zoomLevels.CITY) {
                    }
                });
            
            boroughsGroup
                .selectAll('path')
                .data(boroughData.features)
                .enter()
                .append('path')
                .attr('class', 'borough-path')
                .attr('d', datum => projectionFunction(datum))
                .attr('fill', datum => boroughToHexColor[datum.properties.boro_name])
                .on("click", function (datum) {
                    if (zoomLevel === zoomLevels.ZIPCODE) {
                        zoomReset();
                    } else if (zoomLevel === zoomLevels.BOROUGH) {
                    } else if (zoomLevel === zoomLevels.CITY) {
                        zoomToPath(datum);
                        zoomLevel = zoomLevels.BOROUGH;
                        moveZipCodesGroupToFront();
                    }
                });
            
            updateVisualizationWithDateData = () => {
                const dateString = dateDropdown.value;
                const crashDateData = crashDateDataByDateString[dateString];
                crashesGroup.selectAll('*').remove();
                if (zoomLevel === zoomLevels.ZIPCODE) {
                } else if (zoomLevel === zoomLevels.BOROUGH) {
                } else if (zoomLevel === zoomLevels.CITY) {
                    const boroughTextData = crashDateDataByDateString.reduce((accumulator,) => {
                    });
                    crashesGroup
                        .selectAll('text')
                        .data(boroughTextData)
                        .enter()
                        .append('text')
                        .attr('class', 'borough-text')
                        .text(datum => `${boroughTextData}`);
                }
                // crashesGroup
                //     .selectAll('circle')
                //     .data(crashDateData)
                //     .enter()
                //     .append('path')
                //     .attr('class', 'crash')
                //     .attr('fill', 'red'); // @todo change this based on whether or not someone was injured
            };
            
            updateDate();
        };
        
        render();
        window.addEventListener('resize', render);
        
    }).catch((error) => {
        console.error(error);
    });
};

visualizationMain();
