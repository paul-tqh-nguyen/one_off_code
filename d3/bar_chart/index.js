
const svg = d3.select('svg');
svg.style('background-color', 'grey');

const svg_height = parseFloat(svg.attr('height'));
const svg_width = parseFloat(svg.attr('width'));

// @hack to work around “URL scheme must be ”http“ or ”https“ for CORS request.”
const data_location = "https://raw.githubusercontent.com/paul-tqh-nguyen/one_off_code/master/d3/bar_chart/location_populations.json"; 

const render = data => {
    const xScale = d3.scaleLinear()
          .domain([0, d3.max(data, d=>d.population)]);
    
    console.log(xScale.domain());
    
    svg.selectAll('rect').data(data)
        .enter()
        .append('rect')
        .attr('width', 300)
        .attr('height', 300);
};

d3.json(data_location)
    .then(data => {
        data = data.map(datum => {
            return {
                population: parseFloat(datum.PopTotal) * 1000,
                location: datum.Location_Populations,
            };
        });
        render(data);
    }).catch(err => {
        console.error(err);
        return;
    });
