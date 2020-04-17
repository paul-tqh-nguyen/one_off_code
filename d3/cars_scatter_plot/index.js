
const svg = d3.select('svg');
svg.style('background-color', 'white');

const svg_height = parseFloat(svg.attr('height'));
const svg_width = parseFloat(svg.attr('width'));

const data_location = "https://raw.githubusercontent.com/paul-tqh-nguyen/one_off_code/master/d3/cars_scatter_plot/cars_data.csv";

const render = data => {
    const getDatumMPG = datum => datum.mpg;
    const getDatumCylinders = datum => datum.cylinders;
    const getDatumDisplacement = datum => datum.displacement;
    const getDatumHorsePower = datum => datum.horsepower;
    const getDatumWeight = datum => datum.weight;
    const getDatumAcceleration = datum => datum.acceleration;
    const getDatumYear = datum => datum.year;
    const getDatumOrigin = datum => datum.origin;
    const getDatumName = datum => datum.Name;
    
    const getXValue = getDatumWeight;
    const getYValue = getDatumAcceleration;

    const xAxisLabel = 'Weight';
    const yAxisLabel = 'Acceleration';
    
    const chartTitle = 'Car Comparison';
    
    const margin = {
        top: 80,
        bottom: 100,
        left: 120,
        right: 30,
    };
    const circleRadius = 10;
    
    const innerWidth = svg_width - margin.left - margin.right;
    const innerHeight = svg_height - margin.top - margin.bottom;
    
    const xScale = d3.scaleLinear()
          .domain(d3.extent(data, getXValue))
          .range([0, innerWidth])
          .nice();
    
    const yScale = d3.scaleLinear()
          .domain(d3.extent(data, getYValue))
          .range([0, innerHeight])
          .nice();
    
    const barChartGroup = svg.append('g')
          .attr('transform', `translate(${margin.left}, ${margin.top})`);

    const barChartTitle = barChartGroup.append('text')
          .attr('class', 'chart-title')
          .text(chartTitle)
          .attr('text-anchor', 'middle')
          .attr('x', innerWidth / 2)
          .attr('y', -margin.top / 3);
    
    const yAxis = d3.axisLeft(yScale)
          .tickSize(-innerWidth)
          .tickPadding(20);
    const yAxisGroup = barChartGroup.append('g')
          .call(yAxis);
    yAxisGroup.selectAll('.domain').remove();
    yAxisGroup.append('text') // Y-xaxix label
        .attr('class','axis-label')
        .attr('fill', 'black')
        .attr('y', -margin.left / 2)
        .attr('x', -innerHeight / 2)
        .attr('text-anchor', 'middle')
        .attr('transform', `rotate(-90)`)
        .text(yAxisLabel);
    
    const xAxis = d3.axisBottom(xScale)
          .tickSize(-innerHeight)
          .tickPadding(20);
    const xAxisGroup = barChartGroup.append('g')
          .call(xAxis)
          .attr('transform', `translate(0, ${innerHeight})`);
    xAxisGroup.selectAll('.domain').remove();// remove ticks
    xAxisGroup.append('text') // X-axis label
        .attr('class','axis-label')
        .attr('fill', 'black')
        .attr('y', margin.bottom * 0.75)
        .attr('x', innerWidth / 2)
        .text(xAxisLabel);

    // display data
    barChartGroup.selectAll('rect').data(data)
        .enter()
        .append('circle')
        .attr('cx', datum => xScale(getXValue(datum)))
        .attr('cy', datum => yScale(getYValue(datum)))
        .attr('r', circleRadius);

};

d3.csv(data_location)
    .then(data => {
        data = data.map(datum => {
            return {
                mpg: parseFloat(datum.mpg),
                cylinders: parseFloat(datum.cylinders),
                displacement: parseFloat(datum.displacement),
                horsepower: parseFloat(datum.horsepower),
                weight: parseFloat(datum.weight),
                acceleration: parseFloat(datum.acceleration),
                year: parseInt(datum.year),
                origin: datum.origin,
                name: datum.name,
            };
        });
        render(data);
    }).catch(err => {
        console.error(err);
        return;
    });
