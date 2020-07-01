
const choroplethMain = () => {
    
    const geoJSONLocation = './data/processed_data.geojson';

    const plotContainer = document.getElementById('main-display');
    const svg = d3.select('#choropleth-svg');
    const landMassesGroup = svg.append('g').attr('id', 'land-masses-group');
    const toolTipGroup = svg.append('g').attr('id', 'tool-tip-group');
    
    const paddingAmount = 0;
    
    const toolTipFontSize = 10;
    const toolTipTextPadding = toolTipFontSize;
    const toolTipBackgroundColor = 'red';
    const toolTipTransitionTime = 500;

    d3.json(geoJSONLocation).then(data => {
        const redraw = () => {
            svg
                .attr('width', `${plotContainer.clientWidth}px`)
                .attr('height', `${plotContainer.clientHeight}px`);
            const svgWidth = parseFloat(svg.attr('width'));
            const svgHeight = parseFloat(svg.attr('height'));
            
            const projection = d3.geoMercator()
                  .fitExtent([[paddingAmount, paddingAmount], [svgWidth-paddingAmount, svgHeight-paddingAmount]], data);
            const projectionFunction = d3.geoPath().projection(projection);
            
            const updateToolTip = (x, y, datum) => {
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
                const toolTipX = x + 10;
                const toolTipY = y + 10;
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
                                .style('transition', 'fill-opacity 1.0s')
                                .style('fill-opacity', 0.5);
                            d3.select(this)
                                .style('transition', 'fill-opacity 0.5s')
                                .style('fill-opacity', 1);
                        }
                        d3.select(this).raise();
                        const [mouseX, mouseY] = d3.mouse(this);
                        updateToolTip(mouseX, mouseY, datum);
                    })
                    .on('mouseout', () => {
                        landMassesGroup
                            .selectAll('path')
                            .style('transition', 'fill-opacity 0.5s')
                            .style('fill-opacity', 1);
                        toolTipGroup.selectAll('*').remove();
                    })
                    .attr('d', datum => projectionFunction(datum));
            });
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
