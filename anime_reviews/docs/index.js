
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
    const newElement = childTag === 'svg' ? document.createElementNS('http://www.w3.org/2000/svg', childTag) : document.createElement(childTag);
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
    /* 

scatterPlotData looks like this:
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
    'xMaxValue': 100,
    'yMinValue': 0,
    'yMaxValue': 250,
    'xAxisTitle': 'Rank',
    'yAxisTitle': 'Scores',
}

This returns a re-render function, but does not actually call the re-render function initially.

*/
    // @todo make sure all the attributes of scatterPlotData are used
    
    /* Visualization Aesthetic Parameters */
    
    const margin = {
        top: 80,
        bottom: 80,
        left: 100,
        right: 30,
    };

    const innerLineOpacity = 0.1; // @todo move this to css

    /* Visualization Initialization */

    removeAllChildNodes(container);
    const shadowContainer = createNewElement('div');
    container.append(shadowContainer);
    const shadow = shadowContainer.attachShadow({mode: 'open'});

    const shadowStyleElement = createNewElement('style', {innerHTML: `

:host {
  position: relative;
  width: inherit;
  height: inherit;
}

.scatter-plot-container {
  position: absolute;
  top: 0px;
  bottom: 0px;
  left: 0px;
  right: 0px;
  margin: 0px;
}

`});
    
    shadow.append(shadowStyleElement);
    
    const scatterPlotContainer = createNewElement('div', {classes: ['scatter-plot-container']});
    shadow.append(scatterPlotContainer);
    
    const svg = d3.select(scatterPlotContainer).append('svg');
    const scatterPlotGroup = svg
          .append('g')
          .attr('id', 'scatter-plot-group');
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
            .text(scatterPlotData.yAxisTitle);
        
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
            .text(scatterPlotData.xAxisTitle);

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
    
    return render;
};

/********/
/* Main */
/********/

[
    './result_analysis/rank_0_summary.json',
    // './result_analysis/rank_1_summary.json',
    // './result_analysis/rank_2_summary.json',
    // './result_analysis/rank_3_summary.json',
    // './result_analysis/rank_4_summary.json',
    // './result_analysis/rank_5_summary.json',
    // './result_analysis/rank_6_summary.json',
    // './result_analysis/rank_7_summary.json',
    // './result_analysis/rank_8_summary.json',
    // './result_analysis/rank_9_summary.json',
].forEach(jsonFile => {
    d3.json(jsonFile)
        .then(summaryData => {
            const userScatterPlotContainer = createNewElement('div', {classes: ['user-scatter-plot-container']});
            document.querySelector('body').append(userScatterPlotContainer);
            const userScatterPlotData = {
                'pointSetLookup': {
                    'users': Object.entries(summaryData.user_data).map(([userId, userData]) => Object.assign(userData, {'id': userId})),
                },
                'xAccessor': datum => datum.example_count,
                'yAccessor': datum => datum.mean_mse_loss,
                'toolTipHTMLGenerator': datum => `
<p>User Id: ${datum.id}</p>
<p>Total MSE Loss: ${datum.total_mse_loss}</p>
<p>Mean MSE Loss: ${datum.mean_mse_loss}</p>
<p>Example Count: ${datum.example_count}</p>
`,
                'pointCSSClassAccessor': pointSetName => 'user-scatter-plot-point',
                'xMinValue': 0,
                'xMaxValue': Math.max(...Object.values(summaryData.user_data).map(datum => datum.example_count)) + 10,
                'yMinValue': 0,
                'yMaxValue': Math.max(...Object.values(summaryData.user_data).map(datum => datum.mean_mse_loss)) + 10,
                'xAxisTitle': 'Example count',
                'yAxisTitle': 'Mean MSE Loss',
            };
            const redrawUserScatterPlot = addScatterPlot(userScatterPlotContainer, userScatterPlotData);
            redrawUserScatterPlot();
            
            const animeScatterPlotContainer = createNewElement('div', {classes: ['anime-scatter-plot-container']});
            document.querySelector('body').append(animeScatterPlotContainer);
            const animeScatterPlotData = {
                'pointSetLookup': {
                    'animes': Object.entries(summaryData.anime_data).map(([animeId, animeData]) => Object.assign(animeData, {'id': animeId})),
                },
                'xAccessor': datum => datum.example_count,
                'yAccessor': datum => datum.mean_mse_loss,
                'toolTipHTMLGenerator': datum => `
<p>Anime Id: ${datum.id}</p>
<p>Total MSE Loss: ${datum.total_mse_loss}</p>
<p>Mean MSE Loss: ${datum.mean_mse_loss}</p>
<p>Example Count: ${datum.example_count}</p>
`,
                'pointCSSClassAccessor': pointSetName => 'anime-scatter-plot-point',
                'xMinValue': 0,
                'xMaxValue': Math.max(...Object.values(summaryData.anime_data).map(datum => datum.example_count)) + 10,
                'yMinValue': 0,
                'yMaxValue': Math.max(...Object.values(summaryData.anime_data).map(datum => datum.mean_mse_loss)) + 10,
                'xAxisTitle': 'Example count',
                'yAxisTitle': 'Mean MSE Loss',
            };
            const redrawAnimeScatterPlot = addScatterPlot(animeScatterPlotContainer, animeScatterPlotData);
            redrawAnimeScatterPlot();
            
            window.addEventListener('resize', () => {
                redrawUserScatterPlot();
                redrawAnimeScatterPlot();
            });
        }).catch(err => {
            console.error(err.message);
            return;
        });
});
