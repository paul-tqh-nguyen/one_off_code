
const svg = d3.select('svg');
svg.style('background-color', 'grey');

const svg_height = parseFloat(svg.attr('height'));
const svg_width = parseFloat(svg.attr('width'));

// @hack to work around “URL scheme must be ”http“ or ”https“ for CORS request.”
const data_location = "https://raw.githubusercontent.com/paul-tqh-nguyen/one_off_code/master/d3/bar_chart/location_populations.json"; 

const render = data => {
    const xScale = d3.scaleLinear()
          .domain([0, d3.max(data, d=>d.population)])
          .range([0, svg_width]);
    
    const yScale = d3.scaleBand()
          .domain(data.map(d=>d.location))
          .range([0, svg_height]);
    
    svg.selectAll('rect').data(data)
        .enter()
        .append('rect')
        .attr('width', d => xScale(d.population))
        .attr('y', d => yScale(d.location))
        .attr('height', yScale.bandwidth());
};

d3.json(data_location)
    .then(data => {
        data = data.map(datum => {
            return {
                population: parseFloat(datum.PopTotal) * 1000,
                location: datum.Location,
            };
        });
        render(data);
    }).catch(err => {
        console.error(err);
        return;
    });
