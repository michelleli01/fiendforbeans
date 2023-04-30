import { useEffect, useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import { Typography } from '@mui/material';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Button from '@mui/material/Button';
import LinearProgress, {
  LinearProgressProps,
} from '@mui/material/LinearProgress';
import { Bean } from '../types';
import logo from '../static/images/logo.png';

function importAll(r: __WebpackModuleApi.RequireContext) {
  let images: { [key: string]: any } = {};
  r.keys().forEach((item, index) => {
    images[item.replace('./', '')] = r(item);
  });
  return images;
}

const images = importAll(
  require.context(
    './../static/images/roaster_logos',
    false,
    /\.(png|jpe?g|svg|webp)$/
  )
);

// const mockBean = {
//   bean_info: {
//     id: 69,
//     name: 'Ethiopia Guracho',
//     roaster: 'Sey Coffee',
//     dollars_per_ounce: 3,
//     origin: 'Ethiopia',
//     roaster_link: 'https://seycoffee.com',
//     roast: 'Light',
//     review:
//       'Floral-driven, deep-toned. Lavender, dark chocolate, bing cherry, amber, tangerine zest in aroma and cup. Sweetly tart structure with balanced, juicy acidity; plush, satiny mouthfeel. Resonant finish with notes of lavender and dark chocolate. ',
//   },
//   score: 96,
// };

// const mockResponse = Array(10).fill(mockBean);

const SearchResultsPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [recommended, setRecommended] = useState<Bean[] | undefined>(undefined);
  const [query, setQuery] = useState<string>('');

  const getRecommendedCoffees = (query: string, roast: string) => {
    const searchParams = new URLSearchParams({
      flavor_prof: query,
      roast_value: roast,
    }).toString();

    console.log(searchParams);

    fetch('/beans?' + searchParams)
      .then((response) => response.json())
      .then((data) => {
        setRecommended(data);
        console.log(data);
      });
  };

  useEffect(() => {
    const query = searchParams.get('q');
    const roast = searchParams.get('roast');
    if (query && roast) getRecommendedCoffees(query, roast);
  }, [searchParams]);

  recommended?.forEach((coffee) => console.log(coffee.score));

  return (
    <Box style={{ margin: '2%' }}>
      <Box style={{ display: 'flex', justifyContent: 'space-between' }}>
        <Link to='/' style={{ textDecoration: 'none', color: 'brown' }}>
          <img src={logo} style={{ width: '300px' }} alt='fiendforbeans logo' />
        </Link>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <TextField
            style={{ width: '300px' }}
            label='search...'
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && query.length !== 0) {
                setSearchParams({
                  q: query,
                  roast: searchParams.get('roast') || '',
                });
              }
            }}
          />
          <Button
            variant='contained'
            onClick={() => {
              if (query.length !== 0)
                setSearchParams({
                  q: query,
                  roast: searchParams.get('roast') || '',
                });
            }}
          >
            Search
          </Button>
        </Box>
      </Box>
      <Box
        style={{
          display: 'flex',
          justifyContent: 'space-evenly',
          flexWrap: 'wrap',
          rowGap: '30px',
          margin: '5% 2%',
        }}
      >
        {recommended === undefined ? (
          <Box sx={{ marginTop: '300px' }}>
            <Typography variant='body1'>Loading results...</Typography>
          </Box>
        ) : (
          recommended.map((coffee) => (
            <Card
              variant='outlined'
              style={{
                width: '45%',
                padding: '1%',
                boxShadow: '0px 5px 10px 0px rgba(0, 0, 0, 0.5)',
              }}
            >
              <Box
                style={{ display: 'flex', alignItems: 'center', margin: '1%' }}
              >
                <Box>
                  <a href={coffee.bean_info.roaster_link}>
                    <CardMedia
                      component='img'
                      sx={{ width: 150 }}
                      image={
                        images[`${coffee.bean_info.roaster}.webp`]
                          ? images[`${coffee.bean_info.roaster}.webp`]
                          : undefined
                      }
                      alt='roaster logo'
                    />
                  </a>
                </Box>

                <CardContent sx={{ width: '100%' }}>
                  <Typography component='div' variant='h5'>
                    {coffee.bean_info.name} - {coffee.bean_info.roaster}
                  </Typography>
                  <Typography variant='body1' color='text.secondary'>
                    {coffee.bean_info.review}
                  </Typography>
                </CardContent>
              </Box>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant='subtitle1'>Score:</Typography>
                <LinearProgressWithLabel
                  variant='determinate'
                  value={coffee.score * 100}
                />
                <Typography variant='body2'>
                  Roast: {coffee.bean_info.roast} | Price: $
                  {coffee.bean_info.dollars_per_ounce}
                  /oz
                </Typography>
              </Box>
            </Card>
          ))
        )}
      </Box>
    </Box>
  );
};

function LinearProgressWithLabel(
  props: LinearProgressProps & { value: number }
) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center' }}>
      <Box sx={{ width: '100%', mr: 1 }}>
        <LinearProgress variant='determinate' {...props} />
      </Box>
      <Box sx={{ minWidth: 35 }}>
        <Typography variant='body2' color='text.secondary'>{`${Math.round(
          props.value
        )}%`}</Typography>
      </Box>
    </Box>
  );
}

export default SearchResultsPage;
