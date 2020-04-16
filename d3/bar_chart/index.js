
const svg = d3.select('svg');
svg.style('background-color', 'white');

const svg_height = parseFloat(svg.attr('height'));
const svg_width = parseFloat(svg.attr('width'));

// @hack to work around “URL scheme must be ”http“ or ”https“ for CORS request.”
const data_location = "https://raw.githubusercontent.com/paul-tqh-nguyen/one_off_code/master/d3/bar_chart/location_populations.json"; 

const render = data => {
    const getDatumPopulation = datum => datum.population;
    const getDatumLocation = datum => datum.location;
    const margin = {
        top: 30,
        bottom: 30,
        left: 80,
        right: 30,
    };
    const innerWidth = svg_width - margin.left - margin.right;
    const innerHeight = svg_height - margin.top - margin.bottom;
    
    const xScale = d3.scaleLinear()
          .domain([0, d3.max(data, getDatumPopulation)])
          .range([0, innerWidth]);
    
    const yScale = d3.scaleBand()
          .domain(data.map(getDatumLocation))
          .range([0, innerHeight])
          .padding(0.1);
    
    const group = svg.append('g')
          .attr('transform', `translate(${margin.left}, ${margin.top})`);
    
    const yAxisGroup = group.append('g')
          .call(d3.axisLeft(yScale));
    
    const xAxisGroup = group.append('g')
          .call(d3.axisBottom(xScale))
          .attr('transform', `translate(0, ${innerHeight})`);
    
    group.selectAll('rect').data(data)
        .enter()
        .append('rect')
        .attr('width', datum => xScale(getDatumPopulation(datum)))
        .attr('y', datum => yScale(getDatumLocation(datum)))
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
