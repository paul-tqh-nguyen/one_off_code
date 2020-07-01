
const choroplethMain = () => {
 
    const geoJSONLocation = './data/processed_data.geojson';

    const plotContainer = document.getElementById('main-display');
    const svg = d3.select('#choropleth-svg');
    const landMassesGroup = svg.append('g').attr('id', 'land-masses-group');

    const paddingAmount = 0;

    d3.json(geoJSONLocation).then(data => {
        const redraw = () => {
            svg
                .attr('width', `${plotContainer.clientWidth}px`)
                .attr('height', `${plotContainer.clientHeight}px`);
            const svgWidth = parseFloat(svg.attr('width'));
            const svgHeight = parseFloat(svg.attr('height'));
            
            const landmassData = data.features;
            
            const projection = d3.geoMercator()
                  .fitExtent([[paddingAmount, paddingAmount], [svgWidth-paddingAmount, svgHeight-paddingAmount]], data);
            const projectionFunction = d3.geoPath().projection(projection);
                
            const landMassSelection = landMassesGroup
                  .selectAll('path')
                  .data(landmassData);
            [landMassSelection, landMassSelection.enter().append('path')].map(selection => {
                selection
                    .attr('class', 'land-mass')
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
