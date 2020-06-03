
const mapMain = () => {
    
    const getJSONLocation = 'usa.geojson'; // 'geodata.geojson'
    
    const plotContainer = document.getElementById('map');
    const svg = d3.select('#map-svg');
    const landMassesGroupTranslateLayer = svg.append('g')
          .attr('id','land-masses-group-translate-layer');
    const landMassesGroupScaleLayer = landMassesGroupTranslateLayer.append('g')
          .attr('id','land-masses-group-scale-layer');
    const projection = d3.geoCylindricalStereographic();

    const landMassColor = '#69b3a2';
    const landMassBorderColor = 'red';

    const redraw = () => {
        
        svg
            .attr('width', `${plotContainer.clientWidth}px`)
            .attr('height', `${plotContainer.clientHeight}px`);
        const svg_width = parseFloat(svg.attr('width'));
        const svg_height = parseFloat(svg.attr('height'));
        
        d3.json(getJSONLocation, data =>{
            landMassesGroupScaleLayer
                .selectAll('path')
                .data(data.features)
                .enter()
    	        .append('path')
                .attr('fill', landMassColor)
                .style('stroke', landMassBorderColor)
                .attr('d', d3.geoPath().projection(projection));
            const landMassesGroupScaleLayerBoundingBox = d3.select('#land-masses-group-scale-layer').node().getBBox();
            const landMassesGroupScaleLayerWidth = landMassesGroupScaleLayerBoundingBox.width;
            const landMassesGroupScaleLayerHeight = landMassesGroupScaleLayerBoundingBox.height;
            const landMassesGroupScaleLayerStretchFactor = Math.min(svg_width / landMassesGroupScaleLayerWidth, svg_height / landMassesGroupScaleLayerHeight);
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
