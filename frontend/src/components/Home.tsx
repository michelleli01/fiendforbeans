import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Input from '@mui/material/Input';
import TextField from '@mui/material/TextField';
import MenuItem from '@mui/material/MenuItem';
import Button from '@mui/material/Button';
import logo from './../static/images/logo.png';
import { Roast } from './../types';

const HomePage = () => {
  const [search, setSearch] = useState<string>('');
  const [roast, setRoast] = useState<Roast>('Light');

  const navigate = useNavigate();

  return (
    <Box
      sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}
    >
      <img
        src={logo}
        style={{ width: '300px', margin: '200px 0px 25px' }}
        alt='fiendforbeans logo'
      />
      <Typography variant='body1'>
        Enter which flavor profiles you enjoy.
      </Typography>
      <Box
        sx={{
          border: '1px solid black',
          padding: '0 20px 10px 20px',
          borderRadius: '30px',
          marginBottom: '30px',
        }}
      >
        <Input
          sx={{ marginTop: '25px', width: '600px' }}
          placeholder='citrus, floral, sweet'
          onChange={(e) => setSearch(e.target.value)}
        />
      </Box>
      <Typography>Select which roast you prefer:</Typography>
      <TextField
        sx={{ minWidth: '200px', marginTop: '10px' }}
        select
        label='Roast'
        defaultValue='Light'
        onChange={(e) => setRoast(e.target.value as Roast)}
      >
        <MenuItem value='Light'>Light</MenuItem>
        <MenuItem value='Medium-Light'>Medium-Light</MenuItem>
        <MenuItem value='Medium'>Medium</MenuItem>
        <MenuItem value='Medium-Dark'>Medium-Dark</MenuItem>
        <MenuItem value='Dark'>Dark</MenuItem>
      </TextField>
      <Button
        sx={{
          borderRadius: '15px',
          padding: '1%',
          width: '150px',
          marginTop: '25px',
        }}
        variant='contained'
        onClick={() => {
          if (search.length !== 0) {
            const params = new URLSearchParams({
              q: search,
              roast,
            }).toString();
            navigate(`/search?${params}`);
          }
        }}
      >
        Search
      </Button>
    </Box>
  );
};

export default HomePage;
