const svg = d3.select('svg');
svg.style('background-color', 'white');

const svg_height = parseFloat(svg.attr('height'));
const svg_width = parseFloat(svg.attr('width'));

const data_location = "https://raw.githubusercontent.com/paul-tqh-nguyen/one_off_code/master/d3/temperature_san_francisco_scatter_plot/temperature_in_san_francisco.csv";

const render = data => {
    const getDatumTemperature = datum => datum.temperature;
    const getDatumTimestamp = datum => datum.timestamp;
    
    const getXValue = getDatumTimestamp;
    const getYValue = getDatumTemperature;

    const xAxisLabel = 'Timestamp';
    const yAxisLabel = 'Temperature';
    
    const chartTitle = 'Temperature in San Francisco';
    
    const margin = {
        top: 80,
        bottom: 100,
        left: 120,
        right: 30,
    };
    const circleRadius = 10;
    
    const innerWidth = svg_width - margin.left - margin.right;
    const innerHeight = svg_height - margin.top - margin.bottom;
    
    const xScale = d3.scaleTime()
          .domain(d3.extent(data, getXValue))
          .range([0, innerWidth]);
    
    const yScale = d3.scaleLinear()
          .domain([0,d3.max(data, getYValue)])
          .range([innerHeight, 0])
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
    yAxisGroup.selectAll('.domain');
    yAxisGroup.append('text') // Y-xaxix label
        .attr('class','axis-label')
        .attr('fill', 'black')
        .attr('y', -margin.left / 2)
        .attr('x', -innerHeight / 2)
        .attr('text-anchor', 'middle')
        .attr('transform', `rotate(-90)`)
        .text(yAxisLabel);
    
    const xAxis = d3.axisBottom(xScale)
          .ticks(15)
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

    const areaGenerator = d3.area()
          .x(datum => xScale(getXValue(datum)))
          .y0(innerHeight)
          .y1(datum => yScale(getYValue(datum)))
          .curve(d3.curveBasis);
    barChartGroup.append('path')
        .attr('d', areaGenerator(data))
        .attr('fill-opacity', 1.0)
        .attr('fill','orange')
        .attr('stroke-width', 5)
        .attr('stroke','red');
    
    barChartGroup.selectAll('circle').data(data)
        .enter()
        .append('circle')
        .attr('cx', datum => xScale(getXValue(datum)))
        .attr('cy', datum => yScale(getYValue(datum)))
        .attr('r', circleRadius);

};

d3.csv(data_location)
    .then(data => {
        console.log(`JSON.stringify(data) ${JSON.stringify(data)}`);
        data = data.map(datum => {
            return {
                timestamp: new Date(datum.timestamp),
                temperature: parseFloat(datum.temperature),
            };
        });
        render(data);
    }).catch(err => {
        console.error(err);
        return;
    });
