
const choroplethMain = () => {
 
    const geoJSONLocation = './data/processed_data.geojson';

    const plotContainer = document.getElementById('main-display');
    const svg = d3.select('#choropleth-svg');
    const landMassesGroup = svg.append('g');
    const projection = d3.geoMercator();
    const projectionFunction = d3.geoPath().projection(projection);

    d3.json(geoJSONLocation).then(data => {
        const redraw = () => {
            svg
                .attr('width', `${plotContainer.clientWidth}px`)
                .attr('height', `${plotContainer.clientHeight}px`);
            const svgWidth = parseFloat(svg.attr('width'));
            const svgHeight = parseFloat(svg.attr('height'));

            const landmassData = data.features;
            
            landMassesGroup
                .selectAll('path')
                .data(landmassData)
                .enter()
    	        .append('path')
                .attr('class', 'landmass')
                .attr('d', datum => projectionFunction(datum));
        };
        redraw();
        window.addEventListener('resize', redraw);
    }).catch((error) => {
        console.error('Could not load geographic data.');
        console.error(error);
    });
};

choroplethMain();

const toggleHelp = () => {
    document.getElementById("help-display").classList.toggle("show");
    document.getElementById("main-display").classList.toggle("show");
};
