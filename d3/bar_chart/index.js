
const svg = d3.select('svg');
svg.style('background-color', 'grey');

const svg_height = parseFloat(svg.attr('height'));
const svg_width = parseFloat(svg.attr('width'));

d3.json("location_populations.json")
    .then(data => {
        data = data.map(datum => {
            datum.population = parseFloat(datum.population) * 1000;
            return datum;
        });
        console.log(data);
    }).catch(err => {
        console.error(err);
        return;
    });
