module.exports = {
  // assumptions: {
  //   setSpreadProperties: true,
  // },
  plugins: [
    'lodash',
    // ['@babel/plugin-proposal-object-rest-spread', { useBuiltIns: true }],
  ],
  presets: [
    [
      '@babel/preset-env',
      {
        modules: false,
      },
    ],
    '@babel/preset-typescript',
    '@babel/preset-react',
  ],
};
