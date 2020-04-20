
const svg = d3.select('svg');
svg.style('background-color', 'grey');

const svg_height = parseFloat(svg.attr('height'));
const svg_width = parseFloat(svg.attr('width'));

const colorScale = d3.scaleOrdinal()
      .domain(['apple', 'lemon'])
      .range(['red', 'yellow']);

const radiusScale = d3.scaleOrdinal()
      .domain(['apple', 'lemon'])
      .range([50, 30]);

const render = (selection, { fruits }) => {
    const circles = selection.selectAll('circle').data(fruits);
    circles
        .enter()
        .append('circle')
        .attr('class', datum => datum.type)
        .attr('cx', (datum, index) => (index * 100) + 60)
        .attr('cy', svg_height / 2)
        .merge(circles)
        .attr('fill', d => colorScale(d.type))
        .attr('r', d => radiusScale(d.type));
    circles
        .exit()
        .remove();
};

const makeFruit = type => ({ type });

let fruits = d3.range(5).map(() => makeFruit('apple'));

render(svg, {fruits});

setTimeout(() => {
    fruits.pop();
    render(svg, {fruits});
}, 1000);

setTimeout(() => {
    fruits.pop();
    fruits[1].type='lemon';
    render(svg, {fruits});
}, 2000);


setTimeout(() => {
    fruits = fruits.filter((datum, index) => (index != 0));
    render(svg, {fruits});
}, 3000);
