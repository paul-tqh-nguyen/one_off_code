
let updateDate;

const visualizationMain = () => {
    
    const boroughJSONLocation = './processed_data/borough.json';
    const zipCodesJSONLocation = './processed_data/zip_code.json';
    const crashDateFilesJSONLocation = './processed_data/crash_date_files.json';

    const plotContainer = document.getElementById('visualization-display');
    const svg = d3.select('#map-svg');

    const boroughsGroup = svg.append('g').attr('id', 'boroughs-group');
    const zipCodesGroup = svg.append('g').attr('id', 'zip-codes-group');

    const dateDropdown = document.getElementById('date-dropdown');
    
    Promise.all([
        d3.json(boroughJSONLocation),
        d3.json(zipCodesJSONLocation),
        d3.json(crashDateFilesJSONLocation),
    ]).then(data => {
        
        const [boroughData, zipCodeData, crashDateFileData]  = data;
        
        svg
	    .attr('width', `${plotContainer.clientWidth}px`)
	    .attr('height', `${plotContainer.clientHeight}px`);

        const svgWidth = parseFloat(svg.attr('width'));
	const svgHeight = parseFloat(svg.attr('height'));

        crashDateFileData.forEach(dateFileData => {
            const {} = dateFileData
            
        });
        updateDate = () => { // @todo write this
        };
        
        const projection = d3.geoMercator().fitExtent([[0, 0], [svgWidth, svgHeight]], boroughData);
        const projectionFunction = d3.geoPath().projection(projection);
        
        boroughsGroup
              .selectAll('path')
              .data(boroughData.features)
              .enter()
              .append('path')
              .attr('class', 'borough-path')
              .attr('d', datum => projectionFunction(datum));
        
        zipCodesGroup
              .selectAll('path')
              .data(zipCodeData.features)
              .enter()
              .append('path')
              .attr('class', 'zip-code-path')
              .attr('d', datum => projectionFunction(datum));
        
    }).catch((error) => {
        console.error(error);
    });
};

visualizationMain();
