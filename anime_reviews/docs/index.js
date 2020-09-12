
/***************/
/* Misc. Utils */
/***************/

const isUndefined = value => value === void(0);

/**********************/
/* HTML Element Utils */
/**********************/

const removeAllChildNodes = (parent) => {
    while (parent.firstChild) {
        parent.removeChild(parent.firstChild);
    }
};

const createNewElement = (childTag, {classes, attributes, innerHTML}={}) => {
    const newElement = document.createElement(childTag);
    if (!isUndefined(classes)) {
        classes.forEach(childClass => newElement.classList.add(childClass));
    }
    if (!isUndefined(attributes)) {
        Object.entries(attributes).forEach(([attributeName, attributeValue]) => {
            newElement.setAttribute(attributeName, attributeValue);
        });
    }
    if (!isUndefined(innerHTML)) {
        newElement.innerHTML = innerHTML;
    }
    return newElement;
};

/***************************/
/* Visualization Utilities */
/***************************/

const addScatterPlot = (container, scatterPlotData) => {
    /* scatterPlotData looks like this:
{
    'pointSetLookup': {
        'ps1': [{'x': 12, 'y': 31, 'name': 'Ingrid'}, {'x': 42, 'y': 25, 'name': 'Jure'}], 
        'ps2': [{'x': 94, 'y': 71, 'name': 'Philip'}, {'x': 76, 'y': 17, 'name': 'Nair'}], 
    },
    'xAccessor': datum => datum.x,
    'yAccessor': datum => datum.y,
    'toolTipHTMLGenerator': datum => `<p>Name: ${datum.name}</p>`,
    'pointCSSClassAccessor': pointSetName => {
        return {
            'ps1': 'ps1-point',
            'ps2': 'ps2-point',
        }[pointSetName];
    },
    'xMinValue': 0,
    'xMaxValue': 100.
}
*/
    // @todo make sure all the attributes of scatterPlotData are used
    
    /* Visualization Aesthetic Parameters */
    
    const margin = {
        top: 80,
        bottom: 80,
        left: 220,
        right: 30,
    };

    const innerLineOpacity = 0.1; // @todo move this to css

    /* Visualization Initialization */

    removeAllChildNodes(container);
    const scatterPlotContainer = createNewElement('div');
    container.append(scatterPlotContainer);
    
    const shadow = scatterPlotContainer.attachShadow({mode: 'open'});
    const svgDomElement = createNewElement('svg');
    shadow.append(svgDomElement);
    
    const svg = d3.select(svgDomElement);
    const scatterPlotGroup = svg.append('g');
    const pointSetGroups = Object.keys(scatterPlotData.pointSetLookup).map(
        (pointSetName) => scatterPlotGroup
            .append('g')
            .attr('id', `point-set-group-${pointSetName}`)
    );
    const scatterPlotTitle = scatterPlotGroup.append('text');
    const xAxisGroup = scatterPlotGroup.append('g');
    const xAxisLabel = xAxisGroup.append('text');
    const yAxisGroup = scatterPlotGroup.append('g');
    const yAxisLabel = yAxisGroup.append('text');
    // @todo add legend

    const tooltipDivDomElement = createNewElement('div');
    const tooltipDiv = d3.select(tooltipDivDomElement); // @todo add tooltip display functionality
    
    const render = () => {
        
        svg
            .attr('width', scatterPlotContainer.clientWidth)
            .attr('height', scatterPlotContainer.clientHeight);
        
        const svgWidth = parseFloat(svg.attr('width'));
        const svgHeight = parseFloat(svg.attr('height'));
        
        const innerWidth = svgWidth - margin.left - margin.right;
        const innerHeight = svgHeight - margin.top - margin.bottom;

        const allPoints = [].concat(...Object.values(scatterPlotData.pointSetLookup));
        
        const xScale = d3.scaleLinear()
              .domain([scatterPlotData.xMinValue, scatterPlotData.xMaxValue])
              .range([0, innerWidth]);
        
        const yScale = d3.scaleLinear()
              .domain([scatterPlotData.xMaxValue, scatterPlotData.xMinValue])
              .range([0, innerHeight]);
        
        scatterPlotGroup.attr('transform', `translate(${margin.left}, ${margin.top})`); // @todo move this to css
        
        scatterPlotTitle // @todo handlle css
            .text('Test Loss vs Model Parameter Count')
            .attr('x', innerWidth * 0.325)
            .attr('y', -10);
        
        const yAxisTickFormat = number => d3.format('.3f')(number);
        yAxisGroup.call(d3.axisLeft(yScale).tickFormat(yAxisTickFormat).tickSize(-innerWidth));
        yAxisGroup.selectAll('.tick line')
            .style('opacity', innerLineOpacity);
        yAxisGroup.selectAll('.tick text')
            .attr('transform', 'translate(-3.0, 0.0)');
        yAxisLabel // @todo move some of this to CSS
            .attr('fill', 'black')
            .attr('transform', 'rotate(-90)')
            .attr('y', -60) // @todo move this to a parameter
            .attr('x', -innerHeight/3) // @todo this seems questionable
            .text('Mean Cross Entropy Loss');
        
        const xAxisTickFormat = number => d3.format('.3s')(number).replace(/G/,'B');
        xAxisGroup.call(d3.axisBottom(xScale).tickFormat(xAxisTickFormat).tickSize(-innerHeight))
            .attr('transform', `translate(0, ${innerHeight})`);
        xAxisGroup.selectAll('.tick line')
            .style('opacity', innerLineOpacity);
        xAxisGroup.selectAll('.tick text') // @todo move this to css
            .attr('transform', 'translate(0.0, 5.0)');
        xAxisLabel // @todo add css
            .attr('fill', 'black')
            .attr('y', margin.bottom * 0.75) // @todo this seems questionable
            .attr('x', innerWidth / 2) // @todo this seems questionable
            .text('Parameter Count');

        Object.entries(scatterPlotData.pointSetLookup).forEach(([pointSetName, points]) => {
            const pointSetGroup = scatterPlotGroup
                  .append('g')
                  .attr('id', `point-set-group-${pointSetName}`);
            pointSetGroup.selectAll('circle').data(points)
                .enter()
                .append('circle')
                .on('mouseover', datum => {
                    // @todo add tooltip functionality
                })
                .on('mouseout', datum => {
                    // @todo add tooltip functionality
                })
                .classed(scatterPlotData.pointCSSClassAccessor(pointSetName), true)
                .attr('cx', datum => xScale(scatterPlotData.xAccessor(datum)))
                .attr('cy', datum => yScale(scatterPlotData.yAccessor(datum)));
        });
    };
};

d3.json('') // @todo add a JSON fille
    .then(data => {
        const render = addScatterPlot();
        const redraw = () => {
            render();
        };
        redraw();
        window.addEventListener('resize', redraw);
    }).catch(err => {
        console.error(err.message);
        return;
    });
