
const choroplethMain = () => {
 
    const geoJSONLocation = './data/processed_data.geojson';

    const plotContainer = document.getElementById('main-display');
    const svg = d3.select('#choropleth-svg');
    const landMassesGroup = svg.append('g');

    const paddingAmount = 30;

    d3.json(geoJSONLocation).then(data => {
        const redraw = () => {
            svg
                .attr('width', `${plotContainer.clientWidth}px`)
                .attr('height', `${plotContainer.clientHeight}px`);
            const svgWidth = parseFloat(svg.attr('width'));
            const svgHeight = parseFloat(svg.attr('height'));

            const landmassData = data.features;

            const projection = d3.geoMercator()
                  .translate([svgWidth / 2, svgHeight / 2]);
            const projectionFunction = d3.geoPath().projection(projection);
                
            const landMassSelection = landMassesGroup
                  .selectAll('path')
                  .data(landmassData);
            [landMassSelection, landMassSelection.enter().append('path')].map(selection => {
                selection
                    .attr('class', 'landmass')
                    .attr('d', datum => projectionFunction(datum));
            });
            
            const landMassesGroupBoundingBox = landMassesGroup.node().getBBox();
            const landMassesGroupWidth = landMassesGroupBoundingBox.width;
            const landMassesGroupHeight = landMassesGroupBoundingBox.height;
            const landMassesGroupX = landMassesGroupBoundingBox.x;
            const landMassesGroupY = landMassesGroupBoundingBox.y;
            const landMassesGroupStretchFactor = Math.min( (svgWidth - 2 * paddingAmount) / landMassesGroupWidth, (svgHeight - 2 * paddingAmount) / landMassesGroupHeight);

            landMassesGroup
                .attr('transform', `scale(${landMassesGroupStretchFactor}) translate(${-landMassesGroupBoundingBox.x + 2 * paddingAmount} ${-landMassesGroupBoundingBox.y + 2 * paddingAmount})`);
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
