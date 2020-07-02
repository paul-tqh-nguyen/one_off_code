
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
    
    d3.json(geoJSONLocation).then(data => {
        const earliestDate = new Date(Date.parse(data.earliestDate));
        const latestDate = new Date(Date.parse(data.latestDate));
        const numberOfDays = 1 + (latestDate - earliestDate) / (1000 * 60 * 60 *24);
        const enumeratedDates = d3.range(0, numberOfDays).map(numberOfDaysPassed => new Date(earliestDate.getDate()+numberOfDaysPassed));
        const timeSlider = d3.sliderTop()
              .min(earliestDate)
              .max(latestDate)
              .step(1000 * 60 * 60 * 24)
              .tickFormat(d3.timeFormat('%m/%d/%Y'))
              .tickValues(enumeratedDates)
              .default(earliestDate);
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
                const toolTipTextLines = [
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
                    .attr('class', datum => datum.properties.salesData ? 'land-mass land-mass-with-purchases' : 'land-mass land-mass-without-purchases')
                    .on('mouseover', function (datum) {
                        if (datum.properties.salesData) {
                            landMassesGroup
                                .selectAll('path')
                                .style('fill-opacity', 0.5);
                            d3.select(this)
                                .style('fill-opacity', 1);
                        }
                        d3.select(this).raise();
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

            timeSlider
                .width(parseFloat(svg.attr('width')) * 0.50)
                .on('onchange', sliderValue => {
                    console.log('\n\n');
                    console.log(`earliestDate.getMonth() ${JSON.stringify(earliestDate.getMonth())}`);
                    console.log(`earliestDate.getDay() ${JSON.stringify(earliestDate.getDay())}`);
                    console.log(`earliestDate.getFullYear() ${JSON.stringify(earliestDate.getFullYear())}`);
                    console.log(`earliestDate ${JSON.stringify(earliestDate)}`);
                    console.log(`latestDate ${JSON.stringify(latestDate)}`);
                    console.log(`timeSlider.value() ${JSON.stringify(timeSlider.value())}`);
                });
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
