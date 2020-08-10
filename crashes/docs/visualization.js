
const sum = inputArray => inputArray.reduce((a, b) => a + b, 0);
const mean = inputArray => sum(inputArray) / inputArray.length;
const capitalizeString  = str => {
    var splitStr = str.toLowerCase().split(' ');
    for (var i = 0; i < splitStr.length; i++) {
        splitStr[i] = splitStr[i].charAt(0).toUpperCase() + splitStr[i].substring(1);
    }
    return splitStr.join(' '); 
};

const hexCompliment = hex => {
    let [r,g,b] = ('rgb(' + (hex = hex.replace('#', '')).match(new RegExp('(.{' + hex.length/3 + '})', 'g')).map(function(l) { return parseInt(hex.length%2 ? l+l : l, 16); }).join(',') + ')')
          .replace(/[^\d,]/g, '').split(',').map(intensity => intensity / 255.0);
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2.0;
    if (max === min) {
        h = s = 0;
    } else {
        const d = max - min;
        s = (l > 0.5 ? d / (2.0 - max - min) : d / (max + min));
        if(max == r && g >= b) {
            h = 1.0472 * (g - b) / d ;
        } else if(max == r && g < b) {
            h = 1.0472 * (g - b) / d + 6.2832;
        } else if(max == g) {
            h = 1.0472 * (b - r) / d + 2.0944;
        } else if(max == b) {
            h = 1.0472 * (r - g) / d + 4.1888;
        }
    }
    h = h / 6.2832 * 360.0 + 0;
    h += 180;
    if (h > 360) {
        h -= 360;
    }
    h /= 360;
    if(s === 0){
        r = g = b = l;
    } else {
        const hue2rgb = (p, q, t) => {
            if(t < 0) t += 1;
            if(t > 1) t -= 1;
            if(t < 1/6)
                return p + (q - p) * 6 * t;
            if(t < 1/2)
                return q;
            if(t < 2/3)
                return p + (q - p) * (2/3 - t) * 6;
            return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1/3);
    }
    r = Math.round(r * 255);
    g = Math.round(g * 255); 
    b = Math.round(b * 255);
    const rgb = b | (g << 8) | (r << 16); 
    return "#" + (0x1000000 | rgb).toString(16).substring(1);
};

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
    const plotContainer = document.getElementById('visualization-display');

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

        const sortedDateStrings = Object.keys(crashDateFileData).sort((dateA, dateB) =>  new Date(dateA) - new Date(dateB));
        sortedDateStrings.forEach(dateString => {
            const optionElement = document.createElement('option');
            optionElement.setAttribute('value', dateString);
            optionElement.innerHTML = dateString.split("T")[0];
            dateDropdown.append(optionElement);
        });
        // crashDateDataByDateString has index of date string -> hour -> borough -> zip code
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

	const tooltip = d3.select('#tooltip')
              .style('opacity', 0);
        
        const render = () => {
            
            boroughsGroup.selectAll('*').remove();
            zipCodesGroup.selectAll('*').remove();
            crashesGroup.selectAll('*').remove();
            
            svg
	        .attr('width', `${plotContainer.clientWidth}px`)
	        .attr('height', `${plotContainer.clientHeight}px`);

            const svgWidth = parseFloat(svg.attr('width'));
	    const svgHeight = parseFloat(svg.attr('height'));

            const projection = d3.geoMercator().fitExtent([[0, 0], [svgWidth, svgHeight]], boroughData);
            const projectionFunction = d3.geoPath().projection(projection);

            const centroidByBorough = boroughData.features.reduce((accumulator, feature) => {
                const [centerX, centerY] = projection(d3.geoCentroid(feature));
                accumulator[feature.properties.boro_name] = {centerY: centerY, centerX: centerX};
                return accumulator;
            }, {});
            const crashCountByZipCode = {};
            
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
                crashesGroup.moveToFront();
                zipCodesGroupTransitionOutTimer = setTimeout(() => {
                    if (zoomLevel === zoomLevels.CITY) {
                        boroughsGroup.moveToFront();
                        crashesGroup.moveToFront();
                    }
                }, zoomTransitionTime*4);
            };
            const moveZipCodesGroupToFront = () => {
                zipCodesGroup
                    .transition()
                    .duration(zoomTransitionTime)
                    .attr('stroke-opacity', 1);
                zipCodesGroup.moveToFront();
                crashesGroup.moveToFront();
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
                .on('mouseover', datum => {
                    if (zoomLevel === zoomLevels.BOROUGH) {
                        const toolTipX = d3.event.pageX;
                        const toolTipY = d3.event.pageY;
                        const zipCode = datum.properties.postalCode;
                        const crashCount = crashCountByZipCode[zipCode];
                        tooltip
                            .style('left', `${toolTipX}px`)
                            .style('top', `${toolTipY}px`)
                            .style('opacity', .9);
                        const tooltipElement = document.getElementById('tooltip');
                        const zipCodeParagraphElement = document.createElement('p');
                        zipCodeParagraphElement.innerHTML = `Zip Code: ${zipCode}`;
                        const collisionCountParagraphElement = document.createElement('p');
                        collisionCountParagraphElement.innerHTML = `Collision Count: ${crashCount}`;
                        tooltipElement.querySelectorAll('*').forEach(child => child.remove());
                        tooltipElement.append(zipCodeParagraphElement);
                        tooltipElement.append(collisionCountParagraphElement);
                    }
                })
                .on('mouseout', datum => {
                    tooltip.style('opacity', 0);
                })
                .on("click", function (datum) {
                    tooltip.style('opacity', 0);
                    if (zoomLevel === zoomLevels.ZIPCODE) {
                        zoomReset();
                    } else if (zoomLevel === zoomLevels.BOROUGH) {
                        zoomToPath(datum);
                        zoomLevel = zoomLevels.ZIPCODE;
                    } else if (zoomLevel === zoomLevels.CITY) {
                    }
                    updateVisualizationWithDateData();
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
                    updateVisualizationWithDateData();
                });
            
            updateVisualizationWithDateData = () => {
                const dateString = dateDropdown.value;
                const crashDateData = crashDateDataByDateString[dateString];
                crashesGroup.selectAll('*').remove();
                if (zoomLevel === zoomLevels.ZIPCODE || zoomLevel === zoomLevels.BOROUGH) {
                    const allCrashesForDate = [];
                    crashDateData.forEach(hourData => {
                        Object.values(hourData).forEach(boroughDataForHour => {
                            Object.values(boroughDataForHour).forEach(crashes => {
                                crashes.forEach(crash => {
                                    [crash.x, crash.y] = projection([crash.LONGITUDE, crash.LATITUDE]);
                                    allCrashesForDate.push(crash);
                                });
                            });
                        });
                    });
                    crashesGroup
                        .selectAll('circle')
                        .data(allCrashesForDate)
                        .enter()
                        .append('circle')
                        .attr('cx', datum => datum.x)
                        .attr('cy', datum => datum.y)
                        .attr('class', 'crash')
                        .attr('fill', datum => hexCompliment(boroughToHexColor[capitalizeString(datum.BOROUGH)]))
                        .on('mouseover', datum => {
                            if (zoomLevel === zoomLevels.ZIPCODE) {
                                const toolTipX = d3.event.pageX;
                                const toolTipY = d3.event.pageY;
                                tooltip
                                    .style('left', `${toolTipX}px`)
                                    .style('top', `${toolTipY}px`)
                                    .style('opacity', .9);
                                const tooltipElement = document.getElementById('tooltip');
                                tooltipElement.querySelectorAll('*').forEach(child => child.remove());
                                Object.entries(datum).forEach(keyValuePair => {
                                    const [key, value] = keyValuePair;
                                    if (!['x', 'y', 'LOCATION', 'LONGITUDE', 'LATITUDE', 'CRASH DATE', 'CRASH HOUR', ].includes(key)) {
                                        if (!['Unspecified', 0].includes(value)) {
                                            const paragraphElement = document.createElement('p');
                                            paragraphElement.innerHTML = `${capitalizeString(key.replace('+', ' '))}: ${capitalizeString(String(value))}`;
                                            tooltipElement.append(paragraphElement);
                                        }
                                    }
                                });
                            }
                        })
                        .on('mouseout', datum => {
                            tooltip.style('opacity', 0);
                        })
                        .moveToFront();
                }
                if (zoomLevel === zoomLevels.BOROUGH) {
                    Object.keys(crashCountByZipCode).forEach(key => {
                        delete crashCountByZipCode[key];
                    });
                    crashDateData.forEach(hourData => {
                        Object.values(hourData).forEach(boroughDataForHour => {
                            Object.entries(boroughDataForHour).forEach(zipCodeToCrashesPair => {
                                const [zipCode, crashes] = zipCodeToCrashesPair;
                                if (!crashCountByZipCode.hasOwnProperty(zipCode)) {
                                    crashCountByZipCode[zipCode] = 0;
                                }
                                crashCountByZipCode[zipCode] += crashes.length;
                            });
                        });
                    });
                }
                if (zoomLevel === zoomLevels.CITY) {
                    const boroughTextData = boroughData.features.reduce((accumulator, feature) => {
                        const borough = feature.properties.boro_name;
                        const {centerX, centerY} = centroidByBorough[borough];
                        accumulator[borough] = {
                            centerX: centerX,
                            centerY: centerY,
                            collisionCount: 0,
                        };
                        return accumulator;
                    }, {});
                    crashDateData.forEach(hourData => {
                        Object.entries(hourData).forEach(boroughToBoroughDataPair => {
                            const [borough, boroughDataForHour] = boroughToBoroughDataPair;
                            Object.values(boroughDataForHour).forEach(crashes => {
                                boroughTextData[capitalizeString(borough)].collisionCount += crashes.length;
                            });
                        });
                    });
                    const visualizationTextDisplayDynamicTextElement = document.getElementById('visualization-text-display-dynamic-text');
                    visualizationTextDisplayDynamicTextElement.innerHTML = '';
                    Object.entries(boroughTextData).map(boroughToDataPair => {
                        const [borough, {collisionCount}] = boroughToDataPair;
                        return `${borough} Collisions: ${collisionCount}`;
                    }).sort().forEach(string => {
                        const paragraphElement = document.createElement('p');
                        paragraphElement.innerHTML = string;
                        visualizationTextDisplayDynamicTextElement.append(paragraphElement);
                    });
                    crashesGroup
                        .selectAll('text')
                        .data(Object.entries(boroughTextData))
                        .enter()
                        .append('text')
                        .attr('x', datum => datum[1].centerX)
                        .attr('y', datum => datum[1].centerY)
                        .html(datum => `${datum[0]}: ${datum[1].collisionCount} Collisions`)
                        .attr('class', function(datum) {
                            datum[1].boundingBoxWidth = d3.select(this).node().getBBox().width + 10;
                            datum[1].boundingBoxHeight = d3.select(this).node().getBBox().height + 10;
                            return 'borough-text';
                        });
                    crashesGroup
                        .selectAll('rect')
                        .data(Object.entries(boroughTextData))
                        .enter()
                        .append('rect')
                        .attr('x', datum => datum[1].centerX - datum[1].boundingBoxWidth / 2)
                        .attr('y', datum => datum[1].centerY - datum[1].boundingBoxHeight * 0.65)
                        .attr('width', datum => datum[1].boundingBoxWidth)
                        .attr('height', datum => datum[1].boundingBoxHeight)
                        .attr('class', 'borough-text-bounding-box');
                    crashesGroup
                        .selectAll('text')
                        .data(Object.entries(boroughTextData))
                        .moveToFront();
                }
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
