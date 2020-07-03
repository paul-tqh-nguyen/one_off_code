const parseTimestamp = (timestampStr) => new Date(new Date(timestampStr).getTime() + (new Date(timestampStr).getTimezoneOffset() * 60 * 1000));

const hexToRgb = (hex) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)];
};

const interpolate = (start, end, floatValue) => {
    return start + Math.pow(floatValue, 0.25) * (end - start);
};

const createColorInterpolator = (startHexColor, endHexColor) => {
    const [startR, startG, startB] = hexToRgb(startHexColor);
    const [endR, endG, endB] = hexToRgb(endHexColor);
    const colorInterpolator = (floatValue) => {
        const r = interpolate(startR, endR, floatValue);
        const g = interpolate(startG, endG, floatValue);
        const b = interpolate(startB, endB, floatValue);
        return `rgb(${r}, ${g}, ${b})`;
    };
    return colorInterpolator;
};

const choroplethMain = () => {
    
    const geoJSONLocation = './data/processed_data.geojson';

    const plotContainer = document.getElementById('main-display');
    const svg = d3.select('#choropleth-svg');
    const landMassesGroup = svg.append('g').attr('id', 'land-masses-group');
    const toolTipGroup = svg.append('g').attr('id', 'tool-tip-group');
    const sliderGroup = svg.append('g').attr('id', 'slider-group');
    const sliderBoundingBox = sliderGroup
          .append('rect');    
    const toolTipFontSize = 10;
    const toolTipTextPadding = toolTipFontSize;
    const toolTipBackgroundColor = 'red';
    const toolTipTransitionTime = 500;
    const toolTipMargin = 10;
    
    const sliderBackgroundColor = 'red';
    const sliderTopMargin = 10;
    const sliderPadding = 20;
    const sliderPeriod = 50;

    const landMassWithoutPurchaseColor = '#cccccc';
    const landMassStartColor = '#eeeeee';
    const landMassEndColor = '#289e00';
    const colorMap = createColorInterpolator(landMassStartColor, landMassEndColor);
    
    d3.json(geoJSONLocation).then(data => {
        const earliestDate = parseTimestamp(new Date(Date.parse(data.earliestDate)));
        const latestDate = parseTimestamp(new Date(Date.parse(data.latestDate)));
        const numberOfDays = 1 + (latestDate - earliestDate) / (1000 * 60 * 60 *24);
        const maximumTotalRevenue = data.maximumTotalRevenue;
        const timeSlider = d3.sliderTop()
              .min(earliestDate)
              .max(latestDate)
              .step(1000 * 60 * 60 * 24)
              .tickFormat(d3.timeFormat('%m/%d/%Y'))
              .tickValues(d3.range(0, numberOfDays))
              .default(earliestDate);
        let timer;
        const redraw = () => {
            svg
                .attr('width', `${window.innerWidth * 0.80}px`)
                .attr('height', `${window.innerHeight * 0.80}px`);
            const svgWidth = parseFloat(svg.attr('width'));
            const svgHeight = parseFloat(svg.attr('height'));
            
            const projection = d3.geoMercator()
                  .fitExtent([[0, 0], [svgWidth, svgHeight]], data);
            const projectionFunction = d3.geoPath().projection(projection);
            
            const updateToolTip = (mouseX, mouseY, datum) => {
                const toolTipBoundingBox = toolTipGroup
                      .append('rect')
                      .style('stroke-width', 1)
                      .style('stroke', 'black')
                      .style('fill', toolTipBackgroundColor);
                const toolTipTextLines = [ // @todo fix this
                    'l1',
                    'l2',
                ];
                const textLinesGroup = toolTipGroup.append('g');
                toolTipTextLines.forEach((textLine, textLineIndex) => {
                    textLinesGroup
                        .append('text')
                        .style('font-size', toolTipFontSize)
                        .attr('class', 'tool-tip-text')
                        .attr('dx', toolTipTextPadding)
                        .attr('dy', `${(1+textLineIndex) * 1.2 * toolTipFontSize + toolTipTextPadding / 4}px`)
                        .html(textLine);
                });
                const textLinesGroupBBox = textLinesGroup.node().getBBox();
                const toolTipBoundingBoxWidth = textLinesGroupBBox.width + 2 * toolTipTextPadding;
                const toolTipBoundingBoxHeight = textLinesGroupBBox.height + toolTipTextPadding;
                const mouseCloserToRight = mouseX > parseFloat(svg.attr('width')) - mouseX;
                const toolTipX = mouseCloserToRight ? toolTipMargin : parseFloat(svg.attr('width')) - toolTipMargin - toolTipBoundingBoxWidth;
                const mouseCloserToBottom = mouseY > parseFloat(svg.attr('height')) - mouseY;
                const toolTipY = mouseCloserToBottom ? toolTipMargin : parseFloat(svg.attr('height')) - toolTipMargin - toolTipBoundingBoxHeight;
                toolTipBoundingBox
                    .attr('x', toolTipX)
                    .attr('y', toolTipY)
                    .attr('width', toolTipBoundingBoxWidth)
                    .attr('height', toolTipBoundingBoxHeight);
                textLinesGroup.selectAll('*')
                    .attr('x', toolTipX)
                    .attr('y', toolTipY);
            };
            
            const landmassData = data.features;
            const landMassesGroupSelection = landMassesGroup
                  .selectAll('path')
                  .data(landmassData);
            [landMassesGroupSelection, landMassesGroupSelection.enter().append('path')].forEach(selection => {
                selection
                    .attr('class', 'land-mass')
                    .attr('fill', datum => datum.properties.salesData ? landMassStartColor : landMassWithoutPurchaseColor)
                    .on('mouseover', function (datum) {
                        landMassesGroup
                            .selectAll('path')
                            .style('fill-opacity', 0.25);
                        d3.select(this)
                            .style('fill-opacity', 1);
                        const [mouseX, mouseY] = d3.mouse(this);
                        updateToolTip(mouseX, mouseY, datum);
                    })
                    .on('mouseout', () => {
                        landMassesGroup
                            .selectAll('path')
                            .style('fill-opacity', 1);
                        toolTipGroup.selectAll('*').remove();
                    })
                    .attr('d', datum => projectionFunction(datum));
            });
            
            const landMassesGroupBoundingBox = landMassesGroup.node().getBBox();
            const landMassesGroupWidth = landMassesGroupBoundingBox.width;
            const landMassesGroupHeight = landMassesGroupBoundingBox.height;
            if (svgWidth > landMassesGroupWidth) {
                svg.attr('width', landMassesGroupWidth);
                landMassesGroup.attr('transform', `translate(${-landMassesGroupBoundingBox.x} 0)`);
            }
            if (svgHeight > landMassesGroupHeight) {
                svg.attr('height', landMassesGroupHeight);
                landMassesGroup.attr('transform', `translate(0 ${-landMassesGroupBoundingBox.y})`);
            }
            
            const relevantSalesDataForDate = (date, salesData) => {
                let currentDate = parseTimestamp(new Date(date));
                let relevantSalesData = null;
                while (currentDate > earliestDate && !relevantSalesData) {
                    const year = currentDate.getFullYear().toString();
                    if (salesData[year]) {
                        const month = (currentDate.getMonth()+1).toString();
                        if (salesData[year][month]) {
                            const day = currentDate.getDate().toString();
                            if (salesData[year][month][day]) {
                                relevantSalesData = salesData[year][month][day];
                            }
                        }
                    }
                    currentDate = new Date(currentDate.getTime() - 1000 * 3600 * 24);
                }
                return relevantSalesData; 
            };
            const updateLandMassFill = (sliderDate) => {
                landMassesGroup
                    .selectAll('path')
                    .data(landmassData)
                    .style('fill', datum => {
                        if (datum.properties.salesData) {
                            const relevantSalesData = relevantSalesDataForDate(sliderDate, datum.properties.salesData);
                            const floatValue = relevantSalesData ? relevantSalesData.AmountPaidToDate / maximumTotalRevenue : 0;
                            // if (relevantSalesData && datum.properties.name === 'Australia') {                                
                            // }
                            return colorMap(floatValue);
                        } else {
                            return landMassWithoutPurchaseColor;
                        }
                    });
            };
            timeSlider
                .width(parseFloat(svg.attr('width')) * 0.50)
                .on('onchange', updateLandMassFill);
            sliderGroup.call(timeSlider);
            sliderGroup
                .attr('transform', `translate(${parseFloat(svg.attr('width')) * 0.25} ${parseFloat(svg.attr('height')) * 0.92})`);
            sliderGroup.selectAll('.tick').remove();
            sliderGroup.selectAll('.handle')
                .attr('d','M -5.5,-5.5 v 11 l 12,0 v -11 z');
            sliderGroup
                .selectAll('.parameter-value')
                .select('text')
                .attr('transform', 'translate(0 13)');
            sliderGroup.selectAll('.track-overlay').remove();
            sliderBoundingBox
                .attr('width', 0)
                .attr('height', 0);
            sliderGroup.select('.slider').raise();
            const sliderTrackInsetBoundingBox = sliderGroup.select('.track-inset').node().getBBox();
            const sliderTrackInsetX = sliderTrackInsetBoundingBox.x;
            const sliderTrackInsetY = sliderTrackInsetBoundingBox.y;
            const sliderTrackInsetWidth = sliderTrackInsetBoundingBox.width;
            const sliderTrackInsetHeight = sliderTrackInsetBoundingBox.height;
            sliderBoundingBox
                .style('stroke-width', 1)
                .style('stroke', 'black')
                .style('fill', sliderBackgroundColor)
                .attr('x', sliderTrackInsetX - sliderPadding)
                .attr('y', sliderTrackInsetY - sliderPadding - sliderTopMargin)
                .attr('width', sliderTrackInsetWidth + 2 * sliderPadding)
                .attr('height', sliderTrackInsetHeight + 2 * sliderPadding + sliderTopMargin);
            sliderGroup.select('.slider').raise();
            
            const stopTimer = () => {
                clearInterval(timer);
            };
            const startTimer = () => {
                clearInterval(timer);
                timer = setInterval(() => {
                    if (timeSlider.value().getTime() >= latestDate.getTime()) {
                        clearInterval(timer);
                    } else {
                        timeSlider.value(new Date(timeSlider.value().getTime() + 1000 * 3600 * 24));
                        updateLandMassFill(timeSlider.value());
                    }
                }, sliderPeriod);
            };
            startTimer();
            
        };
        redraw();
        window.addEventListener('resize', redraw);
    }).catch((error) => {
        console.error(error);
    });
};

choroplethMain();

const toggleHelp = () => {
    document.getElementById("help-display").classList.toggle("show");
    document.getElementById("main-display").classList.toggle("show");
};
