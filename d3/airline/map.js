
const mapMain = () => {
    
    const getJSONLocation = './data/processed_data.geojson';
    
    const plotContainer = document.getElementById('map');
    const svg = d3.select('#map-svg');
    const landMassesGroupTranslateLayer = svg.append('g')
          .attr('id','land-masses-group-translate-layer');
    const landMassesGroupScaleLayer = landMassesGroupTranslateLayer.append('g')
          .attr('id','land-masses-group-scale-layer');
    const projection = d3.geoMercator();

    svg.style('background-color', '#aadaff');
    const landMassColor = '#b8d8b4';
    const landMassBorderColor = 'red';
    const flightPathBorderColor = '#9000ff';

    const paddingAmount = 10;

    const redraw = () => {
        svg
            .attr('width', `${plotContainer.clientWidth}px`)
            .attr('height', `${plotContainer.clientHeight}px`);
        const svg_width = parseFloat(svg.attr('width'));
        const svg_height = parseFloat(svg.attr('height'));
        
        d3.json(getJSONLocation, data =>{
            const concat = (a, b) => a.concat(b);
            const isNumber = obj => obj !== undefined && typeof(obj) === 'number' && !isNaN(obj);
            const landmassLongLatPairs = data.features
                  .filter(datum => datum.properties['information-type'] === "landmass")
                  .map(datum => datum.geometry.coordinates.map(coordLists => coordLists[0]).reduce(concat, []))
                  .map(coordList => (coordList.length == 2 && isNumber(coordList[0]) && isNumber(coordList[1])) ? [coordList] : coordList)
                  .reduce(concat, []);
            const landmassLongs = landmassLongLatPairs.map(pair => pair[0]);
            const landmassLats = landmassLongLatPairs.map(pair => pair[1]);
            const landmassMinLong = Math.min(...landmassLongs);
            const landmassMinLat = Math.min(...landmassLats);
            const landmassMaxLong = Math.max(...landmassLongs);
            const landmassMaxLat = Math.max(...landmassLats);

            landMassesGroupScaleLayer
                .selectAll('path')
                .data(data.features.filter(datum => datum.properties['information-type'] === "landmass"))
                .enter()
    	        .append('path')
                .attr('fill', landMassColor)
                .style('stroke', landMassBorderColor)
                .style('stroke-width', 0.25)
                .attr('d', datum => d3.geoPath().projection(projection)(datum));

            const landMassesGroupScaleLayerBoundingBox = d3.select('#land-masses-group-scale-layer').node().getBBox();
            const landMassesGroupScaleLayerWidth = landMassesGroupScaleLayerBoundingBox.width;
            const landMassesGroupScaleLayerHeight = landMassesGroupScaleLayerBoundingBox.height;
            const landMassesGroupScaleLayerStretchFactor = Math.min( (svg_width - 2 * paddingAmount) / landMassesGroupScaleLayerWidth, (svg_height - 2 * paddingAmount) / landMassesGroupScaleLayerHeight);
            
            landMassesGroupScaleLayer
                .selectAll('path')
                .data(data.features
                      .filter(datum => datum.properties['information-type'] === "flight_path")
                      .filter(datum =>
                              datum.geometry.coordinates[0][0] >= landmassMinLong && 
                              datum.geometry.coordinates[0][1] >= landmassMinLat && 
                              datum.geometry.coordinates[1][0] >= landmassMinLong && 
                              datum.geometry.coordinates[1][1] >= landmassMinLat && 
                              datum.geometry.coordinates[0][0] <= landmassMaxLong && 
                              datum.geometry.coordinates[0][1] <= landmassMaxLat && 
                              datum.geometry.coordinates[1][0] <= landmassMaxLong && 
                              datum.geometry.coordinates[1][1] <= landmassMaxLat
                             ))
                .enter()
    	        .append('path')
                .attr('fill', 'none')
                .style('stroke', flightPathBorderColor)
                .style('stroke-width', 1.5/landMassesGroupScaleLayerStretchFactor)
                .style('opacity', 0.1)
                .attr('d', datum => d3.geoPath().projection(projection)(datum));
            
            landMassesGroupScaleLayer
                .attr('transform', `scale(${landMassesGroupScaleLayerStretchFactor})`);
            
            const landMassesGroupTranslateLayerBoundingBox = d3.select('#land-masses-group-translate-layer').node().getBBox();
            const landMassesGroupTranslateLayerWidth = landMassesGroupTranslateLayerBoundingBox.width;
            const landMassesGroupTranslateLayerHeight = landMassesGroupTranslateLayerBoundingBox.height;
            const landMassesGroupTranslateLayerX = landMassesGroupTranslateLayerBoundingBox.x;
            const landMassesGroupTranslateLayerY = landMassesGroupTranslateLayerBoundingBox.y;
            landMassesGroupTranslateLayer
                .attr('transform', `translate(${-landMassesGroupTranslateLayerX + svg_width / 2 - landMassesGroupTranslateLayerWidth / 2} ${-landMassesGroupTranslateLayerY + svg_height / 2 - landMassesGroupTranslateLayerHeight / 2})`);

        });
    };
    
    redraw();
    window.addEventListener('resize', redraw);
};

mapMain();
