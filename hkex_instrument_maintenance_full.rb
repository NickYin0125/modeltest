# frozen_string_literal: true

# Standalone HKEX Instrument Maintenance engine and OMD folder adapter.
# Combines the V2 domain engine and V3 folder adapter without runtime
# dependencies on the original versioned source files.

require "date"
require "bigdecimal"
require "json"
require "set"
require "time"
require "digest"

module HKEX
  module InstrumentMaintenance
    VERSION = "4.1-naming-v2"

    class Error < StandardError; end
    class ValidationError < Error; end
    class ReviewRequired < Error; end
    class CalendarMismatch < Error; end
    class OfficialDataRequired < Error; end
    class GenerationBlocked < Error; end

    module Value
      module_function

      def keys(hash)
        hash.each_with_object({}) { |(key, value), out| out[key.to_sym] = value }
      end

      def deep_keys(value)
        case value
        when Hash then value.each_with_object({}) { |(k, v), out| out[k.to_sym] = deep_keys(v) }
        when Array then value.map { |v| deep_keys(v) }
        else value
        end
      end

      def date(value)
        value.is_a?(Date) ? value : Date.parse(value.to_s)
      end

      def optional_date(value)
        value.nil? || value == "" ? nil : date(value)
      end

      def time(value)
        value.nil? || value == "" ? nil : (value.is_a?(Time) ? value : Time.parse(value.to_s))
      end

      def month(value)
        date = date(value.to_s.length == 7 ? "#{value}-01" : value)
        Date.new(date.year, date.month, 1)
      end

      def decimal(value)
        result = value.is_a?(BigDecimal) ? value : BigDecimal(value.to_s)
        raise ValidationError, "non-finite decimal" unless result.finite?
        result
      end

      def decimal_string(value)
        decimal(value).to_s("F").sub(/\.0+\z/, "").sub(/(\.\d*?)0+\z/, "\\1")
      end

      def iso(value)
        value.respond_to?(:iso8601) ? value.iso8601 : value
      end

      def orderbook_id(value)
        value.is_a?(String) && value.match?(/\A\d+\z/) ? value.to_i : value
      end
    end

    module Enums
      UNDERLYING_TYPES = {
        stock: "S", currency: "C", fixed_income: "I", energy_power: "E",
        commodity: "A", metal: "M"
      }.freeze

      INSTRUMENT_GROUPS = {
        future: 4, call_american: 6, put_american: 7,
        call_european: 22, put_european: 23,
        straddle: 170, strangle: 171, synthetic_future: 172,
        tailor_made_combo: 250, exchange_rate: 254, payment_currency: 255
      }.merge((201..223).to_h { |value| [:"time_spread_#{value}", value] }).freeze

      FINANCIAL_PRODUCTS = {
        option: "O", forward: "F", future: "U", fra: "A", cash: "C",
        payment: "P", exchange_rate: "E", interest_rate_swap: "S", not_available: ""
      }.freeze

      PUT_OR_CALL = { undefined: 0, call: 1, put: 2 }.freeze
      REFERENCE_STATUS = { active: 1, suspended: 2 }.freeze
      RUNTIME_STATUS = { active: 1, suspended: 2, delisted: 4 }.freeze
      TRADABLE = { yes: 1, no: 2 }.freeze
      LEG_SIDE = { buy: 0, sell: 1 }.freeze

      module_function

      def underlying_type(value)
        enum(value, UNDERLYING_TYPES, "UnderlyingType")
      end

      def instrument_group(value)
        return nil if value.nil?
        return INSTRUMENT_GROUPS.key(value) if INSTRUMENT_GROUPS.value?(value)

        normalized = value.to_s.strip.downcase.gsub(/[-\s]+/, "_").to_sym
        return normalized if INSTRUMENT_GROUPS.key?(normalized)

        numeric = Integer(value.to_s, 10) rescue nil
        return INSTRUMENT_GROUPS.key(numeric) if numeric && INSTRUMENT_GROUPS.value?(numeric)

        raise ValidationError, "unknown InstrumentGroup #{value.inspect}"
      end

      def financial_product(value)
        return :not_available if value.nil? || value.to_s.strip.empty?
        enum(value, FINANCIAL_PRODUCTS, "FinancialProduct")
      end

      def put_or_call(value)
        enum(value, PUT_OR_CALL, "PutOrCall")
      end

      def reference_status(value)
        enum(value, REFERENCE_STATUS, "ReferenceInstrumentStatus")
      end

      def runtime_status(value)
        enum(value, RUNTIME_STATUS, "RuntimeInstrumentStatus")
      end

      def tradable(value)
        enum(value, TRADABLE, "Tradable")
      end

      def leg_side(value)
        enum(value, LEG_SIDE, "LegSide")
      end

      def boolean(value, name)
        return false if value.nil? || value == false || value.to_s == "0"
        return true if value == true || value.to_s.downcase == "true"
        return true if %w[Y YES 1].include?(value.to_s.upcase)
        return false if %w[N NO FALSE 0].include?(value.to_s.upcase)
        raise ValidationError, "unknown #{name} #{value.inspect}"
      end

      def enum(value, table, name)
        return nil if value.nil?
        return table.key(value) if table.value?(value)
        normalized = value.to_s.strip.downcase.gsub(/[-\s]+/, "_").to_sym
        return normalized if table.key?(normalized)

        numeric = Integer(value.to_s, 10) rescue nil
        return table.key(numeric) if numeric && table.value?(numeric)

        raise ValidationError, "unknown #{name} #{value.inspect}"
      end
    end

    class Underlying
      attr_reader :commodity_code, :commodity_id, :underlying_code, :underlying_type,
                  :base_currency, :underlying_source, :ca_created, :effective_tomorrow,
                  :commodity_name, :decimal_in_underlying_price

      def initialize(attributes)
        h = Value.keys(attributes)
        @commodity_code = h.fetch(:commodity_code).to_s
        @commodity_id = h[:commodity_id]
        @commodity_name = h[:commodity_name]&.to_s
        @decimal_in_underlying_price = h[:decimal_in_underlying_price] && Integer(h[:decimal_in_underlying_price])
        @underlying_code = h.fetch(:underlying_code).to_s
        @underlying_type = Enums.underlying_type(h.fetch(:underlying_type))
        @base_currency = h[:base_currency]&.to_s
        @underlying_source = h[:underlying_source]&.to_s
        @ca_created = Enums.boolean(h[:ca_created], "CACreated")
        @effective_tomorrow = Enums.boolean(h[:effective_tomorrow], "EffectiveTomorrow")
      end

      def ca_created?
        @ca_created
      end

      def to_h
        {
          commodity_code: commodity_code, commodity_id: commodity_id,
          commodity_name: commodity_name, decimal_in_underlying_price: decimal_in_underlying_price,
          underlying_code: underlying_code, underlying_type: underlying_type,
          base_currency: base_currency, underlying_source: underlying_source,
          ca_created: ca_created, effective_tomorrow: effective_tomorrow
        }
      end
    end

    class InstrumentType
      attr_reader :market, :instrument_group

      def initialize(market:, instrument_group:)
        @market = market.to_s
        @instrument_group = Enums.instrument_group(instrument_group)
      end

      def self.derive(instrument_class)
        new(market: instrument_class.market, instrument_group: instrument_class.instrument_group)
      end
    end

    # Boundary adapter for ODP wire rows.  Domain rows are accepted only when
    # `decoded: true`; otherwise integer price fields are scaled exactly once.
    module ODPAdapter
      module_function

      def decode(row, decoded: false)
        h = Value.keys(row)
        return h.merge(odp_decoded: true) if decoded || h[:odp_decoded]
        scales = {
          contract_size: h[:decimal_in_contract_size],
          price_quotation_factor: h[:decimal_in_contract_size],
          strike_price: h[:decimal_in_strike_price],
          tick_size: h[:decimal_in_price]
        }
        scales.each do |field, scale|
          next if h[field].nil?
          raise ValidationError, "missing scale for #{field}" if scale.nil?
          raise ValidationError, "invalid scale for #{field}" unless Integer(scale).between?(0, 18)
          raise ValidationError, "raw #{field} must be an integer" unless h[field].to_s.match?(/\A-?\d+\z/)
          h[field] = Value.decimal(h[field]) / (BigDecimal(10) ** Integer(scale))
        end
        h[:symbol] ||= h.delete(:instrument_code)
        h[:last_trading_date] ||= h.delete(:expiration_date)
        h[:instrument_status] ||= h.delete(:runtime_instrument_status)
        h[:reference_instrument_status] ||= h.delete(:reference_status)
        h.merge(odp_decoded: true)
      end
    end

    class InstrumentClass
      OPTION_GROUPS = %i[call_american put_american call_european put_european].freeze

      attr_reader :instrument_class_id, :instrument_class_key, :instrument_class_name,
                  :exchange, :market, :instrument_group, :modifier, :commodity_code,
                  :contract_size, :price_quotation_factor, :tick_size,
                  :decimal_in_contract_size, :decimal_in_strike_price,
                  :decimal_in_price, :tradable, :base_currency,
                  :settlement_currency_id, :effective_tomorrow, :generation_mode,
                  :attributes

      def initialize(attributes)
        @attributes = Value.keys(attributes)
        @instrument_class_id = @attributes.fetch(:instrument_class_id)
        @instrument_class_key = @attributes.fetch(:instrument_class_key).to_s
        @instrument_class_name = @attributes[:instrument_class_name]&.to_s
        @exchange = @attributes.fetch(:exchange).to_s
        @market = @attributes.fetch(:market).to_s
        @instrument_group = Enums.instrument_group(@attributes.fetch(:instrument_group))
        @modifier = @attributes[:modifier]&.to_s
        @commodity_code = @attributes.fetch(:commodity_code).to_s
        @contract_size = Value.decimal(@attributes.fetch(:contract_size))
        @price_quotation_factor = Value.decimal(@attributes.fetch(:price_quotation_factor))
        @tick_size = Value.decimal(@attributes.fetch(:tick_size))
        @decimal_in_contract_size = Integer(@attributes.fetch(:decimal_in_contract_size))
        @decimal_in_strike_price = Integer(@attributes.fetch(:decimal_in_strike_price))
        @decimal_in_price = Integer(@attributes.fetch(:decimal_in_price))
        @tradable = Enums.tradable(@attributes.fetch(:tradable))
        @base_currency = @attributes[:base_currency]&.to_s
        @settlement_currency_id = @attributes[:settlement_currency_id]
        @effective_tomorrow = Enums.boolean(@attributes[:effective_tomorrow], "EffectiveTomorrow")
        @generation_mode = (@attributes[:generation_mode] || "STANDARD_NEW_SERIES").to_s.downcase.to_sym
        @instrument_class_id = @attributes[:instrument_class_id]
        raise ValidationError, "contract_size must be positive" unless @contract_size > 0
        raise ValidationError, "price_quotation_factor must be positive" unless @price_quotation_factor > 0
        raise ValidationError, "tick_size must be positive" unless @tick_size > 0
        raise ValidationError, "decimal precision cannot be negative" if [@decimal_in_contract_size, @decimal_in_strike_price, @decimal_in_price].any?(&:negative?)
      end

      def class_id
        instrument_class_id
      end

      def weekly?
        ["18", "39", "WEEKLY", "WEEKLY_OPTION", "WEEKLY_HSI"].include?(market.to_s.upcase)
      end

      def instrument_type
        InstrumentType.derive(self)
      end

      def option?
        OPTION_GROUPS.include?(instrument_group)
      end

      def future?
        instrument_group == :future
      end

      def tradable?
        tradable == :yes
      end

      def eligible_for_standard_generation?
        tradable? && generation_mode == :standard_new_series
      end

      def to_h
        attributes.merge(instrument_group: instrument_group, tradable: tradable)
      end
    end

    class Instrument
      attr_reader :orderbook_id, :symbol, :instrument_class_key, :exchange, :market,
                  :instrument_group, :modifier, :commodity_code, :last_trading_date,
                  :effective_last_trading_date, :strike_price, :datetime_first_trading,
                  :datetime_last_trading, :instrument_status, :financial_product,
                  :put_or_call, :contract_size, :price_quotation_factor,
                  :number_of_legs, :vcm_flag, :isin_code, :effective_tomorrow,
                  :reference_instrument_status, :metadata
      attr_reader :instrument_class_id, :decimal_in_contract_size,
                  :decimal_in_strike_price, :decimal_in_price

      def initialize(attributes)
        h = Value.keys(attributes)
        @orderbook_id = Value.orderbook_id(h[:orderbook_id])
        @symbol = h[:symbol]&.to_s
        raise ValidationError, "Symbol must be 1..32 printable ASCII bytes" unless @symbol && @symbol.bytesize.between?(1, 32) && @symbol.match?(/\A[!-~]+\z/)
        @instrument_class_id = h[:instrument_class_id]
        @instrument_class_key = h.fetch(:instrument_class_key).to_s
        @exchange = h.fetch(:exchange).to_s
        @market = h.fetch(:market).to_s
        @instrument_group = Enums.instrument_group(h.fetch(:instrument_group))
        @modifier = h[:modifier]&.to_s
        @commodity_code = h[:commodity_code]&.to_s
        @last_trading_date = Value.optional_date(h[:last_trading_date])
        @effective_last_trading_date = Value.optional_date(h[:effective_last_trading_date])
        @strike_price = h[:strike_price].nil? ? nil : Value.decimal(h[:strike_price])
        @datetime_first_trading = Value.time(h[:datetime_first_trading])
        @datetime_last_trading = Value.time(h[:datetime_last_trading])
        @instrument_status = Enums.runtime_status(h.fetch(:instrument_status, :active))
        @reference_instrument_status = h[:reference_instrument_status] && Enums.reference_status(h[:reference_instrument_status])
        @financial_product = Enums.financial_product(h.fetch(:financial_product))
        @put_or_call = Enums.put_or_call(h.fetch(:put_or_call, :undefined))
        @contract_size = h[:contract_size].nil? ? nil : Value.decimal(h[:contract_size])
        @price_quotation_factor = h[:price_quotation_factor].nil? ? nil : Value.decimal(h[:price_quotation_factor])
        raise ValidationError, "instrument size and quotation factor must be positive" if [@contract_size, @price_quotation_factor].compact.any? { |value| value <= 0 }
        @number_of_legs = Integer(h.fetch(:number_of_legs, 1))
        @vcm_flag = h[:vcm_flag]
        raise ValidationError, "VCMFlag must be 0 or 1" unless @vcm_flag.nil? || [0, 1, "0", "1"].include?(@vcm_flag)
        @isin_code = h[:isin_code]&.to_s
        @effective_tomorrow = Enums.boolean(h[:effective_tomorrow], "EffectiveTomorrow")
        @decimal_in_contract_size = h[:decimal_in_contract_size] && Integer(h[:decimal_in_contract_size])
        @decimal_in_strike_price = h[:decimal_in_strike_price] && Integer(h[:decimal_in_strike_price])
        @decimal_in_price = h[:decimal_in_price] && Integer(h[:decimal_in_price])
        @metadata = Value.deep_keys(h.fetch(:metadata, {}))
      end

      def self.from(value)
        value.is_a?(self) ? value : new(value)
      end

      def active_for?(target_date)
        target = Value.date(target_date)
        return false unless instrument_status == :active
        return false if reference_instrument_status == :suspended
        effective_expiry = effective_last_trading_date || last_trading_date
        return false if effective_expiry && effective_expiry < target
        return false if datetime_first_trading && datetime_first_trading.to_date > target
        true
      end

      def active_at?(context, expiry: effective_last_trading_date || last_trading_date)
        return false unless instrument_status == :active
        return false if reference_instrument_status == :suspended
        return false if expiry && expiry < context.target_trading_date
        return false if datetime_first_trading && datetime_first_trading > context.target_open_time
        return false if datetime_last_trading && datetime_last_trading < context.target_open_time
        true
      end

      def identity_key
        # Distinct combinations can have identical expiry/strike but different legs.
        return [instrument_class_key, :combination, orderbook_id || symbol] if combination?
        [
          instrument_class_key, last_trading_date || effective_last_trading_date,
          strike_price && Value.decimal_string(strike_price), put_or_call,
          financial_product, modifier
        ]
      end

      def combination?
        %i[straddle strangle synthetic_future tailor_made_combo].include?(instrument_group) ||
          instrument_group.to_s.start_with?("time_spread_") || number_of_legs > 1
      end

      def projection
        {
          "orderbook_id" => orderbook_id,
          "symbol" => symbol,
          "instrument_class_key" => instrument_class_key,
          "last_trading_date" => last_trading_date&.iso8601,
          "effective_last_trading_date" => effective_last_trading_date&.iso8601,
          "strike_price" => strike_price && Value.decimal_string(strike_price),
          "instrument_status" => instrument_status.to_s
        }.compact
      end

      def full_model
        to_h
      end

      def to_h
        projection.merge(
          "instrument_class_id" => instrument_class_id, "exchange" => exchange, "market" => market,
          "datetime_first_trading" => datetime_first_trading&.iso8601,
          "datetime_last_trading" => datetime_last_trading&.iso8601,
          "instrument_group" => instrument_group.to_s,
          "modifier" => modifier, "commodity_code" => commodity_code,
          "financial_product" => financial_product.to_s,
          "put_or_call" => put_or_call.to_s, "contract_size" => contract_size && Value.decimal_string(contract_size),
          "price_quotation_factor" => price_quotation_factor && Value.decimal_string(price_quotation_factor),
          "number_of_legs" => number_of_legs, "vcm_flag" => vcm_flag,
          "isin_code" => isin_code, "effective_tomorrow" => effective_tomorrow,
          "reference_instrument_status" => reference_instrument_status&.to_s,
          "decimal_in_contract_size" => decimal_in_contract_size,
          "decimal_in_strike_price" => decimal_in_strike_price,
          "decimal_in_price" => decimal_in_price,
          "metadata" => metadata
        ).compact
      end

      attr_reader :number_of_legs
    end

    class CombinationLeg
      attr_reader :combo_orderbook_id, :leg_orderbook_id, :leg_side, :leg_ratio

      def initialize(attributes)
        h = Value.keys(attributes)
        @combo_orderbook_id = Value.orderbook_id(h.fetch(:combo_orderbook_id))
        @leg_orderbook_id = Value.orderbook_id(h.fetch(:leg_orderbook_id))
        @leg_side = Enums.leg_side(h.fetch(:leg_side))
        @leg_ratio = Value.decimal(h.fetch(:leg_ratio))
        raise ValidationError, "leg_ratio must be a positive int32" unless @leg_ratio.frac.zero? && @leg_ratio.between?(1, 2_147_483_647)
      end

      def self.from(value)
        value.is_a?(self) ? value : new(value)
      end

      def to_h
        {
          combo_orderbook_id: combo_orderbook_id, leg_orderbook_id: leg_orderbook_id,
          leg_side: leg_side, leg_ratio: Value.decimal_string(leg_ratio)
        }
      end
    end

    class RunContext
      attr_reader :as_of_timestamp, :source_trading_date, :target_trading_date, :calendar_mode
      attr_reader :target_open_time

      def initialize(as_of_timestamp:, source_trading_date:, target_trading_date: nil,
                     calendar_mode: :hkex_real, target_open_time: "09:15:00+08:00")
        @as_of_timestamp = as_of_timestamp.is_a?(Time) ? as_of_timestamp : Time.parse(as_of_timestamp.to_s)
        @source_trading_date = Value.date(source_trading_date)
        @target_trading_date = target_trading_date && Value.date(target_trading_date)
        @calendar_mode = calendar_mode.to_s.downcase.to_sym
        @target_open_time = target_open_time.is_a?(Time) ? target_open_time : Time.parse("#{@target_trading_date || @source_trading_date}T#{target_open_time}")
        unless %i[hkex_real test_synthetic].include?(@calendar_mode)
          raise ValidationError, "unknown calendar_mode #{calendar_mode.inspect}"
        end
      end

      def synthetic?
        calendar_mode == :test_synthetic
      end

      def with_target(target)
        self.class.new(
          as_of_timestamp: as_of_timestamp,
          source_trading_date: source_trading_date,
          target_trading_date: target,
          calendar_mode: calendar_mode,
          target_open_time: target_open_time.strftime("%H:%M:%S%:z")
        )
      end

      def with_target_open_time(value)
        self.class.new(
          as_of_timestamp: as_of_timestamp,
          source_trading_date: source_trading_date,
          target_trading_date: target_trading_date,
          calendar_mode: calendar_mode,
          target_open_time: value
        )
      end
    end

    class TradingCalendar
      attr_reader :holidays

      def initialize(holidays: [], official_monthly_expiries: {}, official_weekly_expiries: {},
                     allow_calendar_override: false, provenance: nil, coverage_from: nil, coverage_to: nil,
                     holiday_completeness: nil, calendar_override_evidence: nil)
        @holidays = holidays.map { |value| Value.date(value) }.to_set
        @official_monthly_expiries = normalize_dates(official_monthly_expiries)
        @official_weekly_expiries = normalize_dates(official_weekly_expiries)
        @allow_calendar_override = allow_calendar_override
        @calendar_override_evidence = calendar_override_evidence
        @provenance = provenance
        @coverage_from = coverage_from && Value.date(coverage_from)
        @coverage_to = coverage_to && Value.date(coverage_to)
        @holiday_completeness = holiday_completeness
      end

      def trading_day?(date)
        date = Value.date(date)
        ![0, 6].include?(date.wday) && !holidays.include?(date)
      end

      def validate_target!(context)
        return unless context.calendar_mode == :hkex_real
        target = context.target_trading_date
        raise ValidationError, "target_trading_date is required" unless target
        raise ValidationError, "#{target} is not a trading day" unless trading_day?(target)
        raise ValidationError, "source is not a trading day" unless trading_day?(context.source_trading_date)
        expected = next_trading_day(context.source_trading_date)
        raise ValidationError, "target_trading_date must be next trading day #{expected}" unless target == expected
        if @provenance.to_s.strip.empty? || @coverage_from.nil? || @coverage_to.nil? || @holiday_completeness != true
          raise ValidationError, "real calendar requires provenance and coverage declaration"
        end
        raise ValidationError, "source outside calendar coverage" unless context.source_trading_date.between?(@coverage_from, @coverage_to)
        raise ValidationError, "target outside calendar coverage" unless target.between?(@coverage_from, @coverage_to)
      end

      def next_trading_day(date)
        candidate = Value.date(date) + 1
        candidate += 1 until trading_day?(candidate)
        candidate
      end

      def previous_trading_day(date)
        candidate = Value.date(date) - 1
        candidate -= 1 until trading_day?(candidate)
        candidate
      end

      def first_trading_day_of_week(date)
        date = Value.date(date)
        monday = date - ((date.wday + 6) % 7)
        monday += 1 until trading_day?(monday)
        monday
      end

      def last_trading_day_of_week(date)
        date = Value.date(date)
        friday = date + (5 - date.wday)
        friday -= 1 until trading_day?(friday)
        friday
      end

      def last_trading_day_of_month(month)
        candidate = Date.new(Value.month(month).year, Value.month(month).month, -1)
        candidate -= 1 until trading_day?(candidate)
        candidate
      end

      def second_last_trading_day_of_month(month)
        previous_trading_day(last_trading_day_of_month(month))
      end

      def resolve_monthly_expiry(period, profile_id: nil)
        derived = second_last_trading_day_of_month(period)
        validate_coverage!(derived)
        official = @official_monthly_expiries[[profile_id, Value.month(period)]] ||
                   @official_monthly_expiries[Value.month(period)]
        return derived unless official
        validate_coverage!(official)
        raise CalendarMismatch, "official expiry is not a trading day" unless trading_day?(official)
        return official if official == derived || (@allow_calendar_override && !@calendar_override_evidence.to_s.empty?)

        raise CalendarMismatch, "official monthly expiry #{official} disagrees with derived #{derived}"
      end

      def resolve_weekly_expiry(period, profile_id: nil)
        derived = last_trading_day_of_week(period)
        validate_coverage!(derived)
        official = @official_weekly_expiries[[profile_id, Value.date(period)]] ||
                   @official_weekly_expiries[Value.date(period)]
        return derived unless official
        validate_coverage!(official)
        raise CalendarMismatch, "official expiry is not a trading day" unless trading_day?(official)
        return official if official == derived || (@allow_calendar_override && !@calendar_override_evidence.to_s.empty?)

        raise CalendarMismatch, "official weekly expiry #{official} disagrees with derived #{derived}"
      end

      private

      def validate_coverage!(date)
        return if @coverage_from.nil? || @coverage_to.nil?
        raise CalendarMismatch, "calendar date #{date} outside declared coverage" unless date.between?(@coverage_from, @coverage_to)
      end

      def normalize_dates(hash)
        hash.each_with_object({}) do |(key, value), out|
          normalized_key = key.is_a?(Array) ? [key[0], key[1].to_s.length == 7 ? Value.month(key[1]) : Value.date(key[1])] :
                           (key.is_a?(Date) ? key : (key.to_s.length == 7 ? Value.month(key) : Value.date(key)))
          out[normalized_key] = Value.date(value)
        end
      end
    end

    class ProductProfile
      attr_reader :id, :family, :instrument_type, :expiry_kind, :reference_kind,
                  :strike_family, :effective_from, :effective_to, :source_reference,
                  :source_basis, :algorithm_basis, :generation_mode, :priority

      def initialize(id:, family:, instrument_type:, expiry_kind:, reference_kind: nil,
                     strike_family: nil, effective_from: nil, effective_to: nil,
                     source_reference: nil, source_basis: :hkex_rule,
                     algorithm_basis: :derived_algorithm, generation_mode: :standard_new_series,
                     priority: 0)
        @id = id.to_s
        @family = family.to_sym
        @instrument_type = instrument_type.to_sym
        @expiry_kind = expiry_kind.to_sym
        @reference_kind = reference_kind&.to_sym
        @strike_family = strike_family&.to_sym
        @effective_from = effective_from && Value.date(effective_from)
        @effective_to = effective_to && Value.date(effective_to)
        @source_reference = source_reference
        @source_basis = source_basis.to_sym
        @algorithm_basis = algorithm_basis.to_sym
        @generation_mode = generation_mode.to_s.downcase.to_sym
        @priority = priority
      end

      def effective_on?(date)
        date = Value.date(date)
        (!effective_from || date >= effective_from) && (!effective_to || date <= effective_to)
      end
    end

    class RuleCatalog
      Entry = Struct.new(:profile, :matcher, keyword_init: true)

      def initialize(entries: nil)
        @entries = entries || default_entries
      end

      def resolve(instrument_class, underlying, target_date)
        candidates = @entries.select do |entry|
          entry.profile.effective_on?(target_date) && entry.matcher.call(instrument_class, underlying)
        end
        return nil if candidates.empty?
        top = candidates.select { |e| [e.profile.priority, e.profile.effective_from || Date.new(1900, 1, 1)] == candidates.map { |x| [x.profile.priority, x.profile.effective_from || Date.new(1900, 1, 1)] }.max }
        raise ValidationError, "ambiguous product profiles" if top.length > 1
        candidates.max_by { |entry| [entry.profile.priority, entry.profile.effective_from || Date.new(1900, 1, 1)] }.profile
      end

      private

      def default_entries
        stock_option = ->(klass) { %i[call_american put_american].include?(klass.instrument_group) }
        index_option = ->(klass) { %i[call_european put_european].include?(klass.instrument_group) }
        future = ->(klass) { klass.future? }
        hsi = ->(klass, underlying) do
          underlying.underlying_type == :stock &&
            [klass.commodity_code, underlying.underlying_code].compact.any? { |code| code.to_s.upcase == "HSI" }
        end
        # Market is the ODP product-family identity. Modifier is retained for
        # CA semantics and must never route a class into a weekly rule.
        weekly_stock = ->(klass) { ["18", "145", "WEEKLY", "WEEKLY_STOCK"].include?(klass.market.to_s.upcase) }
        weekly_hsi = ->(klass) { ["39", "WEEKLY_HSI", "WEEKLY_INDEX"].include?(klass.market.to_s.upcase) }
        monthly_stock = ->(klass) { ["20", "143", "MONTHLY", "EQUITY_OPTION"].include?(klass.market.to_s.upcase) }
        stock_future = ->(klass) { ["2", "141", "STOCK_FUTURE"].include?(klass.market.to_s.upcase) }
        hsi_market = ->(klass) { ["34", "HSI", "HSI_FUTURE", "HSI_OPTION"].include?(klass.market.to_s.upcase) }

        [
          Entry.new(profile: ProductProfile.new(
            id: "HKCC_HSI_WEEKLY_OPTION", family: :hsi_weekly_option,
            instrument_type: :option, expiry_kind: :hsi_weekly,
            reference_kind: :official_index_close, strike_family: :hsi_weekly,
            effective_from: "2026-07-27", source_reference: "HKFE-035/26", priority: 100
          ), matcher: ->(klass, underlying) { index_option.call(klass) && hsi.call(klass, underlying) && weekly_hsi.call(klass) }),
          Entry.new(profile: ProductProfile.new(
            id: "HKCC_HSI_MONTHLY_OPTION", family: :hsi_monthly_option,
            instrument_type: :option, expiry_kind: :hsi_monthly,
            reference_kind: :hsi_closing_quotation, strike_family: :hsi_monthly,
            source_reference: "HKFE-SIO_CS", priority: 100
          ), matcher: ->(klass, underlying) { index_option.call(klass) && hsi.call(klass, underlying) && hsi_market.call(klass) && !weekly_hsi.call(klass) }),
          Entry.new(profile: ProductProfile.new(
            id: "SEOCH_STOCK_OPTION_WEEKLY", family: :seo_weekly_stock_option,
            instrument_type: :option, expiry_kind: :stock_weekly,
            reference_kind: :underlying_security_close, strike_family: :stock_group_a,
            effective_from: "2026-07-27", source_reference: "SEHK-EQD/06/26", priority: 50
          ), matcher: ->(klass, underlying) { underlying.underlying_type == :stock && stock_option.call(klass) && weekly_stock.call(klass) }),
          Entry.new(profile: ProductProfile.new(
            id: "SEOCH_STOCK_OPTION_MONTHLY", family: :seo_monthly_stock_option,
            instrument_type: :option, expiry_kind: :stock_monthly,
            reference_kind: :underlying_security_close, strike_family: :stock_group_b,
            source_reference: "SEHK-Appendix-B", priority: 10
          ), matcher: ->(klass, underlying) { underlying.underlying_type == :stock && stock_option.call(klass) && monthly_stock.call(klass) }),
          Entry.new(profile: ProductProfile.new(
            id: "HKCC_HSI_FUTURE", family: :hsi_future,
            instrument_type: :future, expiry_kind: :hsi_future,
            source_reference: "HKEX-HSI-Futures", priority: 100
          ), matcher: ->(klass, underlying) { future.call(klass) && hsi.call(klass, underlying) && hsi_market.call(klass) }),
          Entry.new(profile: ProductProfile.new(
            id: "HKCC_STOCK_FUTURE", family: :stock_future,
            instrument_type: :future, expiry_kind: :stock_future,
            source_reference: "HKEX-Stock-Futures", priority: 10
          ), matcher: ->(klass, underlying) { underlying.underlying_type == :stock && future.call(klass) && stock_future.call(klass) })
        ]
      end
    end

    class MarketReferenceData
      class Quote
        attr_reader :underlying_code, :quote_type, :quote_date, :contract_month,
                    :value, :source, :is_proxy

        def initialize(attributes)
          h = Value.keys(attributes)
          @underlying_code = h.fetch(:underlying_code).to_s
          @quote_type = h.fetch(:quote_type).to_sym
          @quote_date = Value.date(h.fetch(:quote_date))
          @contract_month = h[:contract_month] && Value.month(h[:contract_month])
          @value = Value.decimal(h.fetch(:value))
          @source = h.fetch(:source).to_s
          @is_proxy = Enums.boolean(h[:is_proxy], "QuoteProxy")
        end
      end

      def initialize(quotes)
        rows = quotes.is_a?(Hash) ? quotes[:quotes] || quotes["quotes"] || [] : quotes
        @quotes = rows.map { |row| row.is_a?(Quote) ? row : Quote.new(row) }
      end

      def find(underlying_code:, quote_type:, quote_date:, contract_month: nil, allow_proxy: false)
        date = Value.date(quote_date)
        month = contract_month && Value.month(contract_month)
        matches = @quotes.select do |item|
          item.underlying_code == underlying_code.to_s &&
            item.quote_type == quote_type.to_sym && item.quote_date == date &&
            item.contract_month == month
        end
        raise OfficialDataRequired, "missing #{quote_type} quote for #{underlying_code} on #{date}" if matches.empty?
        raise ValidationError, "ambiguous #{quote_type} quote for #{underlying_code} on #{date}" if matches.length > 1
        quote = matches.first
        if quote.is_proxy && !allow_proxy
          raise ValidationError, "proxy quote is not allowed for #{quote_type}"
        end

        ReferencePrice.new(
          value: quote.value, reference_price_date: quote.quote_date,
          quote_type: quote.quote_type, contract_month: quote.contract_month,
          source: quote.source, is_proxy: quote.is_proxy
        )
      end
    end

    class ReferencePrice
      attr_reader :value, :reference_price_date, :quote_type, :contract_month, :source, :is_proxy

      def initialize(value:, reference_price_date:, quote_type:, contract_month: nil, source:, is_proxy: false)
        @value = Value.decimal(value)
        @reference_price_date = Value.date(reference_price_date)
        @quote_type = quote_type.to_sym
        @contract_month = contract_month && Value.month(contract_month)
        @source = source.to_s
        @is_proxy = !!is_proxy
      end

      def to_h
        {
          value: Value.decimal_string(value), reference_price_date: reference_price_date.iso8601,
          quote_type: quote_type, contract_month: contract_month&.strftime("%Y-%m"),
          source: source, is_proxy: is_proxy
        }.compact
      end
    end

    class ExpiryDescriptor
      attr_reader :profile_id, :logical_period, :expiration_date, :tenor_bucket, :availability_date

      def initialize(profile_id:, logical_period:, expiration_date:, tenor_bucket:, availability_date:)
        @profile_id = profile_id
        @logical_period = Value.month(logical_period)
        @expiration_date = Value.date(expiration_date)
        @tenor_bucket = tenor_bucket.to_sym
        @availability_date = Value.date(availability_date)
      end
    end

    class ExpiryRule
      QUARTER_MONTHS = [3, 6, 9, 12].freeze

      def initialize(calendar)
        @calendar = calendar
      end

      def generate(profile, target_date)
        case profile.expiry_kind
        when :stock_monthly then monthly(profile, target_date, calendar_count: 3, quarter_count: 3)
        when :stock_future then monthly(profile, target_date, calendar_count: 2, quarter_count: 2)
        when :hsi_future, :hsi_monthly then hsi_monthly(profile, target_date)
        when :stock_weekly, :hsi_weekly then weekly(profile, target_date)
        else raise ValidationError, "unsupported expiry kind #{profile.expiry_kind}"
        end
      end

      def suppressed?(profile, descriptor, target_date)
        return false unless %i[stock_weekly hsi_weekly].include?(profile.expiry_kind)
        monthly_id = profile.expiry_kind == :stock_weekly ? "SEOCH_STOCK_OPTION_MONTHLY" : "HKCC_HSI_MONTHLY_OPTION"
        descriptor.expiration_date == @calendar.resolve_monthly_expiry(Value.month(descriptor.expiration_date), profile_id: monthly_id)
      end

      private

      def spot_month(target_date, profile_id: nil)
        month = Value.month(target_date)
        month = month >> 1 if Value.date(target_date) > @calendar.resolve_monthly_expiry(month, profile_id: profile_id)
        month
      end

      def monthly(profile, target_date, calendar_count:, quarter_count:)
        spot = spot_month(target_date, profile_id: profile.id)
        periods = [spot] + (1..calendar_count).map { |offset| spot >> offset }
        cursor = periods.last >> 1
        while periods.length < 1 + calendar_count + quarter_count
          periods << cursor if QUARTER_MONTHS.include?(cursor.month)
          cursor = cursor >> 1
        end
        descriptors(profile, periods)
      end

      def hsi_monthly(profile, target_date)
        short = monthly_periods(target_date, profile_id: profile.id)
        long = []
        cursor = short.max >> 1
        while long.length < 3
          long << cursor if [6, 12].include?(cursor.month)
          cursor = cursor >> 1
        end
        december = []
        cursor = long.max >> 1
        while december.length < 3
          december << cursor if cursor.month == 12
          cursor = cursor >> 1
        end
        descriptors(profile, short + long + december, short_periods: short)
      end

      def monthly_periods(target_date, profile_id: nil)
        spot = spot_month(target_date, profile_id: profile_id)
        periods = [spot] + (1..3).map { |offset| spot >> offset }
        cursor = periods.last >> 1
        while periods.length < 7
          periods << cursor if QUARTER_MONTHS.include?(cursor.month)
          cursor = cursor >> 1
        end
        periods
      end

      def weekly(profile, target_date)
        date = Value.date(target_date)
        week = @calendar.first_trading_day_of_week(date)
        week = @calendar.first_trading_day_of_week(week + 7) if date > @calendar.resolve_weekly_expiry(week, profile_id: profile.id)
        descriptors(profile, [week, week + 7], weekly: true)
      end

      def descriptors(profile, periods, short_periods: [], weekly: false)
        seen = {}
        periods.map do |period|
          period = weekly ? Value.date(period) : Value.month(period)
          tenor = if short_periods.empty? then :none else (short_periods.include?(period) ? :short : :long) end
          raise ValidationError, "conflicting tenor for #{period}" if seen.key?(period)
          seen[period] = tenor
          expiration = weekly ? @calendar.resolve_weekly_expiry(period, profile_id: profile.id) :
                               @calendar.resolve_monthly_expiry(period, profile_id: profile.id)
          # Availability is the ladder's introduction/current applicability,
          # not a gate tied to the expiry immediately following this row.
          # The target opening state already selects the correct spot ladder.
          availability = @calendar.trading_day?(period) ? Value.date(period) :
                          (weekly ? @calendar.first_trading_day_of_week(period) : Value.date(period))
          ExpiryDescriptor.new(
            profile_id: profile.id, logical_period: period,
            expiration_date: expiration, tenor_bucket: tenor,
            availability_date: availability
          )
        end
      end
    end

    class ValidStrikeGrid
      attr_reader :id, :interval, :anchor, :lower_bound, :source, :evidence_status

      def initialize(id:, interval:, anchor: 0, lower_bound: 0.01,
                     source: :verified_algorithm, evidence_status: :derived_algorithm)
        @id = id.to_s
        @interval = Value.decimal(interval)
        @anchor = Value.decimal(anchor)
        @lower_bound = Value.decimal(lower_bound)
        @source = source.to_sym
        @evidence_status = evidence_status.to_sym
        raise ValidationError, "grid interval must be positive" unless @interval > 0
      end

      def floor(value)
        value = Value.decimal(value)
        @anchor + ((value - @anchor) / @interval).floor * @interval
      end

      def ceil(value)
        value = Value.decimal(value)
        lower = floor(value)
        lower == value ? lower : lower + @interval
      end

      def nearest(value, tie: :lower)
        value = Value.decimal(value)
        lower = floor(value)
        upper = lower + @interval
        (value - lower) <= (upper - value) ? lower : upper
      end

      def predecessors(value, count)
        result = []
        current = floor(value) - @interval
        count.times do
          break if current < @lower_bound
          result << current
          current -= @interval
        end
        result
      end

      def successors(value, count)
        result = []
        current = ceil(value)
        current += @interval if current == Value.decimal(value)
        count.times do
          result << current
          current += @interval
        end
        result
      end

      def between(lower, upper)
        current = ceil(lower)
        result = []
        while current <= upper
          result << current if current >= @lower_bound
          current += @interval
        end
        result
      end
    end

    module StrikeTables
      module_function

      STOCK_GROUP_B = [[0.01, 2, 0.05], [2, 5, 0.10], [5, 10, 0.25], [10, 20, 0.50],
                       [20, 50, 1.00], [50, 200, 2.50], [200, 300, 5.00], [300, Float::INFINITY, 10.00]].freeze
      STOCK_GROUP_A = [[0.01, 2, 0.05], [2, 5, 0.05], [5, 10, 0.10], [10, 20, 0.25],
                       [20, 50, 0.50], [50, 150, 1.00], [150, 300, 2.50], [300, Float::INFINITY, 5.00]].freeze

      def interval(family, value)
        value = Value.decimal(value)
        table = case family
                when :stock_group_a then STOCK_GROUP_A
                when :stock_group_b then STOCK_GROUP_B
                when :hsi_short then [[0, 5000, 50], [5000, 20_000, 100], [20_000, Float::INFINITY, 200]]
                when :hsi_long then [[0, 5000, 100], [5000, 20_000, 200], [20_000, Float::INFINITY, 400]]
                when :hsi_weekly then [[0, 5000, 10], [5000, 20_000, 25], [20_000, Float::INFINITY, 50]]
                else raise ValidationError, "unknown strike family #{family}"
                end
        row = if family.to_s.start_with?("hsi")
                table.find { |low, high, _| (high == Float::INFINITY && value >= low) || (value >= low && value < high) }
              else
                table.find.with_index do |(low, high, _), index|
                  high == Float::INFINITY ? value > low : (index.zero? ? value >= low && value <= high : value > low && value <= high)
                end
              end
        raise ValidationError, "no strike interval for #{value}" unless row
        row[2]
      end
    end

    class StrikeRule
      Result = Struct.new(:strikes, :grid, :coverage_policy, :configuration, keyword_init: true)

      def initialize(grid_configuration: nil, real: false)
        @grid_configuration = grid_configuration
        @real = real
      end

      def generate(profile, descriptor, reference_price, target_date, real: @real)
        family = strike_family(profile, descriptor)
        configured = nil
        grid = nil
        if @grid_configuration
          resolved = @grid_configuration.resolve(profile_id: profile.id, rule_kind: :strike_grid,
                                                  on: target_date, variant: family)
          if resolved.is_a?(Array)
            grid, configured = resolved
          else
            configured = resolved
            grid = configured && (configured.value.is_a?(ValidStrikeGrid) ? configured.value : ValidStrikeGrid.new(**configured.value.transform_keys(&:to_sym)))
          end
        end
        raise ReviewRequired, "approved strike grid required" if real && !configured
        if real && (!configured.approved? || !%i[hkex_rule official_reference derived_algorithm system_policy].include?(configured.evidence_class))
          raise ReviewRequired, "unapproved strike grid #{configured.rule_id}"
        end
        grid ||= grid_for(family, reference_price.value)
        atm = grid.nearest(reference_price.value)
        strikes, policy = case profile.family
                          when :seo_monthly_stock_option
                            lower = piecewise_predecessors(family, atm, 2, grid.lower_bound)
                            upper = piecewise_successors(family, atm, 2)
                            raise ReviewRequired, "STRIKE_LOWER_BOUND_INSUFFICIENT" unless lower.length == 2
                            [lower + [atm] + upper, :minimum_two_each_side]
                          when :seo_weekly_stock_option
                            lower = piecewise_predecessors(family, atm, 20, grid.lower_bound)
                            raise ReviewRequired, "STRIKE_LOWER_BOUND_INSUFFICIENT" unless lower.length == 20
                            [lower + [atm] + piecewise_successors(family, atm, 20), :minimum_twenty_each_side]
                          when :hsi_monthly_option
                            percentage_range(grid, atm, descriptor.tenor_bucket == :short ? "0.10" : "0.20",
                                             descriptor.tenor_bucket == :short ? :hsi_short : :hsi_long,
                                             descriptor.tenor_bucket == :short ? :outward : :nearest)
                          when :hsi_weekly_option
                            percentage_range(grid, atm, "0.10", family, :outward)
                          else
                            [[], :none]
                          end
        Result.new(strikes: strikes, grid: grid, coverage_policy: policy, configuration: configured)
      end

      private

      def strike_family(profile, descriptor)
        return :hsi_short if profile.family == :hsi_monthly_option && descriptor.tenor_bucket == :short
        return :hsi_long if profile.family == :hsi_monthly_option
        profile.strike_family
      end

      def grid_for(family, value)
        interval = StrikeTables.interval(family, value)
        ValidStrikeGrid.new(
          id: "#{family}-#{Value.decimal_string(interval)}",
          interval: interval,
          source: :verified_algorithm,
          evidence_status: :derived_algorithm
        )
      end

      def piecewise_predecessors(family, value, count, lower_bound)
        result = []
        current = Value.decimal(value)
        count.times do
          interval = stepping_interval(family, current, :down)
          current -= interval
          break if current < lower_bound
          result << current
        end
        result.reverse
      end

      def piecewise_successors(family, value, count)
        result = []
        current = Value.decimal(value)
        count.times do
          current += stepping_interval(family, current, :up)
          result << current
        end
        result
      end

      # At a band edge the step in the direction of travel belongs to the
      # band being entered. A tiny decimal probe selects that side of the
      # published inclusive/exclusive boundary without floating point math.
      def stepping_interval(family, value, direction)
        probe = Value.decimal(value) + (direction == :up ? BigDecimal("1e-24") : -BigDecimal("1e-24"))
        StrikeTables.interval(family, probe)
      end

      def percentage_range(grid, atm, percentage, family, mode)
        ratio = BigDecimal("1") - BigDecimal(percentage)
        lower_target = atm * ratio
        upper_target = atm * (BigDecimal("2") - ratio)
        lower = [atm]
        current = atm
        while current > lower_target
          candidate = current - stepping_interval(family, current, :down)
          if candidate < lower_target && mode == :nearest
            lower << candidate if (candidate - lower_target).abs < (current - lower_target).abs
            break
          end
          lower << candidate
          current = candidate
        end
        upper = []
        current = atm
        while current < upper_target
          candidate = current + stepping_interval(family, current, :up)
          if candidate > upper_target && mode == :nearest
            upper << candidate if (candidate - upper_target).abs < (current - upper_target).abs
            break
          end
          upper << candidate
          current = candidate
        end
        [lower.reverse + upper, mode == :outward ? :outward_percentage : :nearest_percentage]
      end
    end

    class ReferencePriceRule
      def initialize(calendar)
        @calendar = calendar
      end

      def resolve(profile, instrument_class, underlying, descriptor, market_data, context)
        case profile.reference_kind
        when :underlying_security_close
          market_data.find(underlying_code: underlying.underlying_code,
                           quote_type: :underlying_security_close,
                           quote_date: context.source_trading_date)
        when :official_index_close
          market_data.find(underlying_code: underlying.underlying_code,
                           quote_type: :official_index_close,
                           quote_date: @calendar.previous_trading_day(context.target_trading_date))
        when :hsi_closing_quotation
          month = hsi_reference_month(context.target_trading_date, profile.id)
          market_data.find(underlying_code: underlying.underlying_code,
                           quote_type: :hkcc_closing_quotation,
                           quote_date: @calendar.previous_trading_day(context.target_trading_date),
                           contract_month: month)
        else
          raise OfficialDataRequired, "reference price rule missing for #{profile.id}"
        end
      end

      private

      def hsi_reference_month(target_date, profile_id)
        month = Value.month(target_date)
        target_date < @calendar.resolve_monthly_expiry(month, profile_id: profile_id) ? month : (month >> 1)
      end
    end

    class CodeResolution
      attr_reader :orderbook_id, :symbol, :source

      def initialize(orderbook_id:, symbol:, source:)
        @orderbook_id = Value.orderbook_id(orderbook_id)
        @symbol = symbol.to_s
        raise ValidationError, "Symbol must be 1..32 printable ASCII bytes" unless @symbol.bytesize.between?(1, 32) && @symbol.match?(/\A[!-~]+\z/)
        @source = source.to_s
      end
    end

    class OfficialCodeResolver
      def initialize(entries = {})
        @entries = entries.each_with_object({}) do |(key, value), out|
          normalized = [key[0].to_s, Value.date(key[1]), key[2].nil? ? nil : Value.decimal_string(key[2])]
          raise ValidationError, "duplicate official code mapping" if out.key?(normalized)
          out[normalized] = value
        end
      end

      def future(instrument_class:, expiration_date:)
        resolve([instrument_class.instrument_class_key, Value.date(expiration_date), nil])
      end

      def option(instrument_class:, expiration_date:, strike_price:)
        resolve([instrument_class.instrument_class_key, Value.date(expiration_date), Value.decimal_string(strike_price)])
      end

      private

      def resolve(key)
        value = @entries[key]
        result = if value.is_a?(CodeResolution)
                   value
                 elsif value.is_a?(Hash)
                   CodeResolution.new(**Value.keys(value))
                 end
        if result
          raise ValidationError, "official OrderBookID must be uint32" unless result.orderbook_id.to_s.match?(/\A\d+\z/) && result.orderbook_id.to_i.between?(0, 4_294_967_295)
          raise OfficialDataRequired, "official code source missing" if result.source.strip.empty?
          return result
        end
        raise OfficialDataRequired, "official instrument code mapping missing for #{key.inspect}"
      end
    end

    # HKATS User's Guide, August 2026, Appendix 1 (pp. 100-104).
    # Root and display precision belong to the product reference configuration;
    # CommodityCode and InstrumentClassKey are NOT human-readable root symbols.
    # Unknown/adjusted/Flexible/legacy strategies are explicitly reference-only.
    class SeriesNaming
      SOURCE = "https://www.hkex.com.hk/Services/Trading/Derivatives/Infrastructure/HKATS/HKATS-User-Guide?sc_lang=en"
      FUTURE_MONTHS = %w[F G H J K M N Q U V X Z].freeze
      CALL_MONTHS = %w[A B C D E F G H I J K L].freeze
      PUT_MONTHS = %w[M N O P Q R S T U V W X].freeze
      KINDS = %i[future monthly_option weekly_option calendar_spread hibor_strip tailor_made_combo reference].freeze

      def initialize(rules:)
        @rules = rules.each_with_object({}) do |(key, raw), out|
          raise ValidationError, "duplicate naming class #{key}" if out.key?(key.to_s)
          rule = Value.keys(raw)
          rule[:kind] = rule.fetch(:kind).to_sym
          raise ValidationError, "unknown naming kind" unless KINDS.include?(rule[:kind])
          raise ValidationError, "naming configuration source required" if rule[:source].to_s.strip.empty?
          if rule[:kind] != :reference
            raise ValidationError, "root must be an explicit 3..5 character HKATS code" unless rule[:root].to_s.match?(/\A[A-Z0-9]{3,5}\z/)
          end
          if %i[monthly_option weekly_option].include?(rule[:kind])
            rule[:strike_decimals] = Integer(rule.fetch(:strike_decimals))
            raise ValidationError, "invalid strike display precision" unless rule[:strike_decimals].between?(0, 12)
          end
          out[key.to_s] = rule.freeze
        end.freeze
      end

      def rule_for(klass)
        rule = @rules.fetch(klass.instrument_class_key) { raise OfficialDataRequired, "naming configuration missing for #{klass.instrument_class_key}" }
        weekly = %w[18 39 84 87 145 WEEKLY WEEKLY_STOCK WEEKLY_HSI WEEKLY_INDEX]
        monthly = %w[16 20 34 38 70 86 143 MONTHLY EQUITY_OPTION HSI_OPTION]
        if (rule[:kind] == :monthly_option && weekly.include?(klass.market)) ||
           (rule[:kind] == :weekly_option && monthly.include?(klass.market))
          raise ValidationError, "naming frequency disagrees with instrument Market"
        end
        if %w[35 37].include?(klass.market) && rule[:kind] != :reference
          raise OfficialDataRequired, "Flexible options require authoritative Symbols"
        end
        rule
      end

      def symbol_for(instrument_class:, contract_month: nil, expiration_date: nil,
                     strike_price: nil, far_month: nil, leg_count: nil, sequence: nil)
        klass = instrument_class
        rule = rule_for(klass)
        kind = rule[:kind]
        if kind == :reference || ![nil, "", "0", "MONTHLY", "WEEKLY"].include?(klass.modifier&.upcase)
          raise OfficialDataRequired, "authoritative Symbol required for adjusted/reference series"
        end
        root = rule[:root]
        month = contract_month && Value.month(contract_month)
        expiry = expiration_date && Value.date(expiration_date)
        case kind
        when :future
          raise ValidationError, "future naming requires FUTURE group" unless klass.future?
          name = root + month_code(month, FUTURE_MONTHS)
        when :monthly_option, :weekly_option
          raise ValidationError, "option naming requires an option group" unless klass.option?
          codes = %i[call_american call_european].include?(klass.instrument_group) ? CALL_MONTHS : PUT_MONTHS
          # A week starting in the previous month is named by its expiry date.
          month = Value.month(expiry) if kind == :weekly_option && expiry
          raise ValidationError, "weekly expiry date required" if kind == :weekly_option && !expiry
          name = root + format_strike(strike_price, rule[:strike_decimals]) + month_code(month, codes)
          name += "W#{expiry.strftime('%d')}" if kind == :weekly_option
        when :calendar_spread
          raise ValidationError, "calendar spread requires a TIME_SPREAD group" unless klass.instrument_group.to_s.start_with?("time_spread_")
          far = far_month && Value.month(far_month)
          raise ValidationError, "near/far contract months must be ordered" unless month && far && month < far
          name = root + month_code(month, FUTURE_MONTHS) + "/" + month_code(far, FUTURE_MONTHS)
        when :hibor_strip
          expected = { "H1S" => 3, "H3S" => 4 }[root]
          raise ValidationError, "HIBOR strip root/leg count mismatch" unless expected && leg_count == expected
          name = root + month_code(month, FUTURE_MONTHS) + leg_count.to_s
        when :tailor_made_combo
          raise ValidationError, "TMC naming requires TAILOR_MADE_COMBO group" unless klass.instrument_group == :tailor_made_combo
          # The exchange assigns this sequence. Never allocate it locally.
          raise OfficialDataRequired, "exchange-assigned TMC sequence required" unless sequence.to_s.match?(/\A\d{1,3}\z/) && sequence.to_i.between?(0, 999)
          name = "TMC_#{root}/#{format('%03d', sequence.to_i)}"
        end
        CodeResolution.new(orderbook_id: nil, symbol: name, source: SOURCE).symbol
      end

      private

      def month_code(month, codes)
        raise ValidationError, "explicit contract_month required" unless month
        "#{codes.fetch(month.month - 1)}#{month.year % 10}"
      end

      def format_strike(value, decimals)
        raise ValidationError, "strike price required" if value.nil?
        strike = Value.decimal(value)
        raise ValidationError, "strike must be positive and exactly representable" unless strike.positive? && strike.round(decimals) == strike
        whole, fraction = strike.to_s("F").split(".")
        decimals.zero? ? whole : "#{whole}.#{fraction.to_s.ljust(decimals, '0')[0, decimals]}"
      end
    end

    # Format-only synthesis is for the test universe. Real OrderBookIDs and
    # Symbols must come from the daily reference map, never from a hash.
    class NamedCodeResolver
      attr_reader :naming

      def initialize(naming:, entries: {}, synthetic: false)
        @naming = naming
        @official = OfficialCodeResolver.new(entries)
        @synthetic = synthetic == true
      end

      def synthetic?; @synthetic end

      def future(instrument_class:, expiration_date:, contract_month: nil)
        resolve(instrument_class, expiration_date, nil, contract_month)
      end

      def option(instrument_class:, expiration_date:, strike_price:, contract_month: nil)
        resolve(instrument_class, expiration_date, strike_price, contract_month)
      end

      def validate_instrument!(instrument, instrument_class:, context:)
        raise GenerationBlocked, "synthetic naming resolver in real mode" if synthetic? && !context.synthetic?
        validate_id!(instrument.orderbook_id)
        rule = naming.rule_for(instrument_class)
        if instrument.combination?
          source = instrument.metadata[:source] || instrument.metadata[:source_reference]
          raise OfficialDataRequired, "combination reference provenance required" if source.to_s.strip.empty?
        end
        if rule[:kind] == :reference || ![nil, "", "0", "MONTHLY", "WEEKLY"].include?(instrument.modifier&.upcase)
          source = instrument.metadata[:source] || instrument.metadata[:source_reference]
          raise OfficialDataRequired, "reference Symbol provenance required for #{instrument.symbol}" if source.to_s.strip.empty?
          return
        end
        # A later effective LTD must not rename an existing contract.
        expiry = instrument.last_trading_date
        month = instrument.metadata[:contract_month] || instrument.metadata[:logical_period]
        if month.nil? && %i[future monthly_option].include?(rule[:kind])
          # An imported authoritative Symbol is already reference data. The ODP
          # model has no contract-month field, so preserve it without guessing.
          source = instrument.metadata[:source] || instrument.metadata[:source_reference]
          return unless source.to_s.strip.empty?
          raise OfficialDataRequired, "contract_month metadata required for #{instrument.symbol}"
        end
        expected = naming.symbol_for(instrument_class: instrument_class, contract_month: month,
          expiration_date: expiry, strike_price: instrument.strike_price,
          far_month: instrument.metadata[:far_contract_month], leg_count: instrument.number_of_legs,
          sequence: instrument.metadata[:exchange_sequence])
        raise ValidationError, "Symbol mismatch: #{instrument.symbol}, expected #{expected}" unless instrument.symbol == expected
      end

      private

      def resolve(klass, expiry, strike, month)
        expected = naming.symbol_for(instrument_class: klass, contract_month: month,
          expiration_date: expiry, strike_price: strike)
        if synthetic?
          period = naming.rule_for(klass)[:kind] == :weekly_option ? Value.month(expiry) : Value.month(month)
          identity = [klass.instrument_class_key, Value.date(expiry).iso8601, period.iso8601, strike && Value.decimal_string(strike)]
          # Stable across Ruby processes. Collisions are rejected by the engine;
          # this ID is TEST_POLICY, never an exchange-assigned identifier.
          id = 1 + Digest::SHA256.hexdigest(JSON.generate(identity))[0, 8].to_i(16) % 4_294_967_295
          return CodeResolution.new(orderbook_id: id, symbol: expected, source: "TEST_POLICY:HKATS_FORMAT")
        end
        result = if strike.nil?
                   @official.future(instrument_class: klass, expiration_date: expiry)
                 else
                   @official.option(instrument_class: klass, expiration_date: expiry, strike_price: strike)
                 end
        raise ValidationError, "official Symbol disagrees with configured convention: #{result.symbol} / #{expected}" unless result.symbol == expected
        result
      end

      def validate_id!(value)
        raise ValidationError, "OrderBookID must be uint32" unless value.to_s.match?(/\A\d+\z/) && value.to_i.between?(0, 4_294_967_295)
      end
    end

    class DeterministicCodeResolver
      def synthetic?; true end
      def future(instrument_class:, expiration_date:)
        code = "#{instrument_class.instrument_class_key}|#{Value.date(expiration_date).strftime('%Y%m%d')}"
        code = "TEST#{Digest::SHA256.hexdigest(code)[0, 28]}" if code.bytesize > 32
        CodeResolution.new(orderbook_id: code, symbol: code, source: "TEST_DETERMINISTIC")
      end

      def option(instrument_class:, expiration_date:, strike_price:)
        code = "#{instrument_class.instrument_class_key}|#{Value.date(expiration_date).strftime('%Y%m%d')}|#{Value.decimal_string(strike_price)}"
        code = "TEST#{Digest::SHA256.hexdigest(code)[0, 28]}" if code.bytesize > 32
        CodeResolution.new(orderbook_id: code, symbol: code, source: "TEST_DETERMINISTIC")
      end
    end

    class Generator
      Result = Struct.new(:candidates, :profile, :blocked_reason, keyword_init: true)

      def initialize(calendar:, catalog:, code_resolver:, reference_rule: nil, strike_rule: nil, rule_configuration: nil)
        @calendar = calendar
        @catalog = catalog
        @code_resolver = code_resolver
        @reference_rule = reference_rule || ReferencePriceRule.new(calendar)
        @rule_configuration = rule_configuration
        @strike_rule = strike_rule || StrikeRule.new(grid_configuration: rule_configuration)
        @expiry_rule = ExpiryRule.new(calendar)
      end

      def generate(instrument_class:, underlying:, market_data:, context:)
        if @code_resolver.respond_to?(:synthetic?) && @code_resolver.synthetic? && !context.synthetic?
          return Result.new(candidates: [], blocked_reason: "synthetic_code_resolver_in_real_mode")
        end
        unless instrument_class.eligible_for_standard_generation?
          return Result.new(candidates: [], blocked_reason: "class_not_tradable_or_generation_disabled")
        end
        if underlying.ca_created?
          return Result.new(candidates: [], blocked_reason: "corporate_action_underlying_requires_official_reference")
        end

        profile = @catalog.resolve(instrument_class, underlying, context.target_trading_date)
        return Result.new(candidates: [], blocked_reason: "official_reference_required_for_unmapped_group") unless profile
        return Result.new(candidates: [], blocked_reason: "adjusted_series_requires_official_reference") if instrument_class.modifier.to_s != "" && instrument_class.modifier.to_s != "0" && instrument_class.modifier.to_s !~ /\A(?:MONTHLY|WEEKLY)\z/i
        if profile.generation_mode != :standard_new_series
          return Result.new(candidates: [], profile: profile, blocked_reason: "profile_generation_mode_#{profile.generation_mode}")
        end

        descriptors = @expiry_rule.generate(profile, context.target_trading_date)
        candidates = []
        identities = {}
        descriptors.each do |descriptor|
          next if @expiry_rule.suppressed?(profile, descriptor, context.target_trading_date)
          if profile.instrument_type == :future
            candidate = build_future(instrument_class, profile, descriptor, context)
            add_unique!(identities, candidates, candidate)
            next
          end

          reference = @reference_rule.resolve(profile, instrument_class, underlying, descriptor, market_data, context)
          strike_result = @strike_rule.generate(profile, descriptor, reference, context.target_trading_date,
                                                real: !context.synthetic?)
          strike_result.strikes.each do |strike|
            candidate = build_option(instrument_class, profile, descriptor, strike, reference, strike_result, context)
            add_unique!(identities, candidates, candidate)
          end
        end
        Result.new(candidates: candidates, profile: profile)
      end

      private

      def build_future(klass, profile, descriptor, context)
        resolution = resolve_code(:future, klass, descriptor)
        metadata = generation_metadata(profile, descriptor, nil, resolution, context)
        # The v3 availability gate is explicit for stock-option month/week
        # series. Futures are already listed across the generated ladder; an
        # availability date must come from the official futures series file,
        # not from the stock-option release convention.
        metadata.delete(:availability_date)
        Instrument.new(
          orderbook_id: resolution.orderbook_id, symbol: resolution.symbol,
          instrument_class_id: klass.instrument_class_id, instrument_class_key: klass.instrument_class_key, exchange: klass.exchange,
          market: klass.market, instrument_group: klass.instrument_group,
          modifier: klass.modifier, commodity_code: klass.commodity_code,
          last_trading_date: descriptor.expiration_date,
          effective_last_trading_date: descriptor.expiration_date,
          instrument_status: :active, reference_instrument_status: :active,
          financial_product: :future,
          put_or_call: :undefined, contract_size: klass.contract_size,
          price_quotation_factor: klass.price_quotation_factor,
          decimal_in_contract_size: klass.decimal_in_contract_size,
          decimal_in_strike_price: klass.decimal_in_strike_price, decimal_in_price: klass.decimal_in_price,
          number_of_legs: 1, effective_tomorrow: true,
          metadata: metadata
        )
      end

      def build_option(klass, profile, descriptor, strike, reference, strike_result, context)
        rounded = strike.round(klass.decimal_in_strike_price)
        raise ReviewRequired, "STRIKE_PRECISION_MISMATCH" unless rounded == strike
        strike = rounded
        resolution = resolve_code(:option, klass, descriptor, strike)
        Instrument.new(
          orderbook_id: resolution.orderbook_id, symbol: resolution.symbol,
          instrument_class_id: klass.instrument_class_id, instrument_class_key: klass.instrument_class_key, exchange: klass.exchange,
          market: klass.market, instrument_group: klass.instrument_group,
          modifier: klass.modifier, commodity_code: klass.commodity_code,
          last_trading_date: descriptor.expiration_date,
          effective_last_trading_date: descriptor.expiration_date,
          strike_price: strike, instrument_status: :active,
          reference_instrument_status: :active,
          financial_product: :option, put_or_call: put_or_call(klass),
          contract_size: klass.contract_size,
          price_quotation_factor: klass.price_quotation_factor,
          decimal_in_contract_size: klass.decimal_in_contract_size,
          decimal_in_strike_price: klass.decimal_in_strike_price, decimal_in_price: klass.decimal_in_price,
          number_of_legs: 1, effective_tomorrow: true,
          metadata: generation_metadata(profile, descriptor, reference, resolution, context).merge(
            strike_grid_id: strike_result.grid.id, coverage_policy: strike_result.coverage_policy,
            rule_configuration: (strike_result.configuration && { rule_id: strike_result.configuration.rule_id, version: strike_result.configuration.version, source_reference: strike_result.configuration.source_reference })
          )
        )
      end

      def generation_metadata(profile, descriptor, reference, resolution, context)
        {
          product_profile_id: profile.id, rule_version: VERSION,
          source_reference: profile.source_reference,
          source_basis: profile.source_basis,
          algorithm_basis: profile.algorithm_basis,
          logical_period: descriptor.logical_period.to_s,
          contract_month: (%i[stock_weekly hsi_weekly].include?(profile.expiry_kind) ? Value.month(descriptor.expiration_date) : descriptor.logical_period).strftime("%Y-%m"),
          tenor_bucket: descriptor.tenor_bucket,
          # Descriptor availability is ladder applicability, while target
          # opening validity is evaluated here. Keep the descriptor for audit
          # and use target as the generated candidate's effective gate.
          availability_date: context.target_trading_date.iso8601,
          descriptor_availability_date: descriptor.availability_date.iso8601,
          reference_price: reference&.to_h,
          code_source: resolution.source,
          target_trading_date: context.target_trading_date.iso8601
        }
      end

      def put_or_call(klass)
        case klass.instrument_group
        when :call_american, :call_european then :call
        when :put_american, :put_european then :put
        else :undefined
        end
      end

      def resolve_code(type, klass, descriptor, strike = nil)
        args = { instrument_class: klass, expiration_date: descriptor.expiration_date }
        args[:strike_price] = strike if type == :option
        args[:contract_month] = descriptor.logical_period if @code_resolver.is_a?(NamedCodeResolver)
        @code_resolver.public_send(type, **args)
      end

      def add_unique!(identities, candidates, candidate)
        key = candidate.identity_key
        raise ValidationError, "duplicate economic identity #{key.inspect}" if identities[key]
        identities[key] = true
        candidates << candidate
      end
    end

    class MaintenanceResult
      attr_reader :instruments, :audit, :reconciliation, :blocked

      def initialize(instruments:, audit:, reconciliation:, blocked:, combination_legs: [], expiry_field: nil)
        @instruments = instruments
        @audit = audit
        @reconciliation = reconciliation
        @blocked = blocked
        @combination_legs = combination_legs
        @expiry_field = expiry_field
      end

      def projection
        raise GenerationBlocked, blocked.map { |row| row["reason_code"] }.join(", ") unless blocked.empty?
        instruments.sort_by { |instrument| [instrument.last_trading_date || Date.new(9999, 12, 31), instrument.symbol.to_s] }
                    .map { |instrument| {
                      "instrument_code" => instrument.symbol,
                      "expiration_date" => ((@expiry_field == :last_trading_date ? instrument.last_trading_date : instrument.effective_last_trading_date || instrument.last_trading_date)&.iso8601)
                    } }
      end

      def diagnostics
        { "blocked" => blocked, "audit" => audit, "reconciliation" => reconciliation }
      end

      def full_model
        { "instruments" => instruments.map(&:full_model), "combination_legs" => @combination_legs.map { |leg| leg.to_h.transform_keys(&:to_s) }, "audit" => audit,
          "reconciliation" => reconciliation, "blocked" => blocked }
      end

      def to_h
        { "instruments" => projection, "audit" => audit, "reconciliation" => reconciliation, "blocked" => blocked }
      end
    end

    class MaintenanceEngine
      def initialize(calendar: TradingCalendar.new, catalog: RuleCatalog.new,
                     code_resolver: OfficialCodeResolver.new({}), scope_mode: :target_only,
                     strike_rule: nil, lifecycle_policy: nil, rule_configuration: nil)
        @calendar = calendar
        @catalog = catalog
        @code_resolver = code_resolver
        @scope_mode = scope_mode.to_sym
        @lifecycle_policy = lifecycle_policy
        @rule_configuration = rule_configuration
        @generator = Generator.new(calendar: calendar, catalog: catalog, code_resolver: code_resolver,
                                   strike_rule: strike_rule, rule_configuration: rule_configuration)
      end

      def maintain(context:, underlyings:, instrument_classes:, instruments:, combination_legs: [],
                   market_data: [], target_underlying_ids: nil, overrides: [], official_reference: nil,
                   mode: :standard)
        context = resolve_context(context)
        if @lifecycle_policy
          if !context.synthetic? && (!@lifecycle_policy.approved? || !%i[hkex_rule official_reference derived_algorithm system_policy].include?(@lifecycle_policy.evidence_class))
            raise ReviewRequired, "lifecycle policy is not approved"
          end
          context = context.with_target_open_time(@lifecycle_policy.target_opening_at(context.target_trading_date))
        end
        @calendar.validate_target!(context)
        if !context.synthetic? && !@lifecycle_policy
          raise ReviewRequired, "approved lifecycle policy required for real snapshot"
        end
        if @lifecycle_policy && !@lifecycle_policy.effective_on?(context.target_trading_date)
          raise ReviewRequired, "lifecycle policy is outside its effective range"
        end
        underlying_map = normalize_underlyings(underlyings)
        classes = instrument_classes.map { |item| item.is_a?(InstrumentClass) ? item : InstrumentClass.new(item) }
        if classes.map(&:instrument_class_key).uniq.length != classes.length
          raise ValidationError, "duplicate instrument class key"
        end
        current = instruments.map { |item| Instrument.from(item) }
        all_class_keys = classes.map(&:instrument_class_key).to_set
        unknown_current = current.reject { |i| all_class_keys.include?(i.instrument_class_key) }
        raise GenerationBlocked, "unknown instrument class foreign key" unless unknown_current.empty?
        legs = combination_legs.map { |item| CombinationLeg.from(item) }
        validate_relationships!(underlying_map, classes, current, [])

        target_ids = target_underlying_ids&.map(&:to_s)&.to_set
        if target_ids
          target_ids = target_ids.map do |id|
            next id if underlying_map.key?(id)
            matches = underlying_map.values.select { |u| [u.underlying_code, u.commodity_id].compact.map(&:to_s).include?(id) }
            raise ValidationError, "unknown or ambiguous target underlying #{id}" unless matches.length == 1
            matches.first.commodity_code
          end.to_set
        end
        requested_mode = mode.to_sym
        raise ValidationError, "unsupported maintenance mode #{mode}" unless %i[standard maintain_existing_only reference_only].include?(requested_mode)
        target_classes = classes.select do |klass|
          next true unless target_ids
          target_ids.include?(klass.commodity_code.to_s)
        end
        target_keys = target_classes.map(&:instrument_class_key).to_set
        class_map = target_classes.to_h { |klass| [klass.instrument_class_key, klass] }
        if @lifecycle_policy
          target_classes.each do |klass|
            raise ReviewRequired, "lifecycle policy market mismatch" unless ["*", klass.market].include?(@lifecycle_policy.market)
            profile = @catalog.resolve(klass, underlying_for(klass, underlying_map), context.target_trading_date)
            permitted = ["*", klass.instrument_class_key, klass.instrument_class_id.to_s, profile&.id]
            raise ReviewRequired, "lifecycle policy product mismatch" unless permitted.include?(@lifecycle_policy.product_id)
          end
        end
        overrides = overrides.map do |raw|
          ov = Value.keys(raw)
          validate_override_selector!(ov, class_map)
          if ov[:instrument_class_id]
            key = class_map.values.find { |klass| klass.instrument_class_id.to_s == ov[:instrument_class_id].to_s }.instrument_class_key
            raise ValidationError, "conflicting override class selectors" if ov[:instrument_class_key] && ov[:instrument_class_key].to_s != key
            ov = ov.merge(instrument_class_key: key)
          end
          ov
        end
        scoped_current = current.select { |instrument| target_keys.include?(instrument.instrument_class_key) }
        untouched_current = current.reject { |instrument| target_keys.include?(instrument.instrument_class_key) }
        ensure_unique_current!(scoped_current)
        market_data = MarketReferenceData.new(market_data)
        audit = []
        blocked = []

        generated = []
        if mode.to_sym == :reference_only
          raise OfficialDataRequired, "reference-only mode requires official_reference" if official_reference.nil? || official_reference.empty?
          allowed_keys = target_classes.map(&:instrument_class_key).map(&:to_s).to_set
          generated = official_reference.map { |item| Instrument.from(item) }
                                      .select { |i| allowed_keys.include?(i.instrument_class_key.to_s) }
          generated.each do |instrument|
            raise OfficialDataRequired, "reference-only source is missing" if instrument.metadata[:source].to_s.empty? && instrument.metadata[:source_reference].to_s.empty?
          end
        end
        target_classes.each do |klass|
          next if klass.generation_mode == :maintain_existing_only
          next if %i[reference_only maintain_existing_only].include?(mode.to_sym)
          suppressed = overrides.find do |ov|
            ov[:action].to_s.upcase == "SUPPRESS_GENERATION" && effective_override?(ov, context.target_trading_date) &&
              ov[:instrument_class_key].to_s == klass.instrument_class_key &&
              %i[orderbook_id instrument_code expiration_date].none? { |field| ov[field] }
          end
          if suppressed
            audit << { "action" => "SUPPRESS_CLASS", "instrument_class_key" => klass.instrument_class_key,
                       "override_source" => suppressed[:source], "reason_code" => suppressed[:reason],
                       "target_trading_date" => context.target_trading_date.iso8601 }
            next
          end
          underlying = underlying_for(klass, underlying_map)
          result = @generator.generate(instrument_class: klass, underlying: underlying,
                                       market_data: market_data, context: context)
          if result.blocked_reason
            blocked << block_record(klass, context, result.blocked_reason, result.profile)
            next
          end
          generated.concat(result.candidates)
        rescue ReviewRequired, OfficialDataRequired, CalendarMismatch => error
          blocked << block_record(klass, context, error.message, nil)
        end

        validate_relationships!(underlying_map, target_classes, generated, [])
        ensure_unique_current!(generated)
        generated = generated.select { |candidate| current_valid?(candidate, class_map, context) }

        # An economic identity that is present but suspended, reference
        # suspended, delisted, or not-yet-live is authoritative state. It may
        # not be resurrected by the minimum generated ladder.
        invalid_existing = scoped_current.reject { |instrument| current_valid?(instrument, class_map, context) }
        protected_identities = invalid_existing.map(&:identity_key).to_set
        generated.reject! { |candidate| protected_identities.include?(candidate.identity_key) }

        generated = apply_generation_suppression(generated, overrides, context, audit)
        survivors = scoped_current.select { |instrument| current_valid?(instrument, class_map, context) }
        survivor_set = survivors.to_set
        scoped_current.each do |instrument|
          next if survivor_set.include?(instrument)
          audit << decision("REMOVE", instrument, context, "target_invalid_or_inactive")
        end
        survivors.each { |instrument| audit << decision("KEEP", instrument, context, "target_valid_existing") }

        final_by_identity = survivors.to_h { |instrument| [instrument.identity_key, instrument] }
        generated.each do |candidate|
          next if final_by_identity.key?(candidate.identity_key)
          final_by_identity[candidate.identity_key] = candidate
          audit << decision("ADD", candidate, context, "minimum_rule_candidate")
        end

        apply_explicit_overrides!(final_by_identity, overrides, class_map, context, audit)
        final = final_by_identity.values
        validate_relationships!(underlying_map, target_classes, final, [])
        validate_final!(final, class_map, context)

        if @scope_mode == :global_merge
          merged = untouched_current + final
          %i[orderbook_id symbol].each do |field|
            ids = merged.map { |instrument| instrument.public_send(field) }.compact
            raise ValidationError, "cross-scope #{field} conflict" unless ids.uniq.length == ids.length
          end
          final = merged
        end
        recon_reference = official_reference
        if @scope_mode == :target_only && official_reference
          allowed = target_classes.map { |k| k.instrument_class_key.to_s }.to_set
          recon_reference = official_reference.select { |item| allowed.include?(Instrument.from(item).instrument_class_key.to_s) }
        end
        output_ids = final.map(&:orderbook_id).to_set
        output_legs = legs.select { |leg| output_ids.include?(leg.combo_orderbook_id) }
        validate_combinations!(final, output_legs, classes.to_h { |klass| [klass.instrument_class_key, klass] })
        reconciliation = reconcile(final, recon_reference, context)
        MaintenanceResult.new(instruments: final, audit: audit, reconciliation: reconciliation, blocked: blocked, combination_legs: output_legs,
                              expiry_field: @lifecycle_policy&.expiry_field)
      end

      private

      def resolve_context(context)
        return context if context.target_trading_date
        context.with_target(@calendar.next_trading_day(context.source_trading_date))
      end

      def normalize_underlyings(values)
        values.each_with_object({}) do |item, out|
          underlying = item.is_a?(Underlying) ? item : Underlying.new(item)
          if out.key?(underlying.commodity_code)
            raise ValidationError, "duplicate CommodityCode #{underlying.commodity_code}"
          end
          out[underlying.commodity_code] = underlying
        end
      end

      def underlying_for(klass, underlying_map)
        underlying = underlying_map[klass.commodity_code] || underlying_map.values.find { |u| u.underlying_code == klass.commodity_code }
        raise ValidationError, "underlying missing for #{klass.instrument_class_key}: #{klass.commodity_code}" unless underlying
        underlying
      end

      def validate_relationships!(underlying_map, classes, instruments, legs)
        classes.each { |klass| underlying_for(klass, underlying_map) }
        class_by_key = classes.to_h { |klass| [klass.instrument_class_key, klass] }
        instruments.each do |instrument|
          klass = class_by_key[instrument.instrument_class_key]
          raise ValidationError, "unknown instrument class foreign key" unless klass
          if klass.option? && instrument.financial_product != :option
            raise ValidationError, "instrument FinancialProduct disagrees with option class"
          end
          if klass.future? && instrument.financial_product != :future
            raise ValidationError, "instrument FinancialProduct disagrees with future class"
          end
          %i[exchange market instrument_group modifier commodity_code].each do |field|
            next if field == :modifier && instrument.modifier.nil?
            raise ValidationError, "instrument #{field} disagrees with class" unless instrument.public_send(field).to_s == klass.public_send(field).to_s
          end
          expected = case klass.instrument_group
                     when :call_american, :call_european then :call
                     when :put_american, :put_european then :put
                     else :undefined
                     end
          raise ValidationError, "instrument PutOrCall disagrees with class" unless instrument.put_or_call == expected
          if klass.option?
            raise ValidationError, "option requires positive strike and LTD" unless instrument.strike_price&.positive? && instrument.last_trading_date
          elsif klass.future?
            raise ValidationError, "future requires LTD and zero/absent strike" unless instrument.last_trading_date && (instrument.strike_price.nil? || instrument.strike_price.zero?)
          end
        end
        known_orderbooks = instruments.map(&:orderbook_id).compact.to_set
        legs.each do |leg|
          raise ValidationError, "combination leg #{leg.leg_orderbook_id} is not present" unless known_orderbooks.include?(leg.leg_orderbook_id)
        end
      end

      def validate_combinations!(instruments, legs, class_map)
        index = instruments.to_h { |instrument| [instrument.orderbook_id, instrument] }
        by_parent = legs.group_by(&:combo_orderbook_id)
        instruments.select(&:combination?).each do |parent|
          members = by_parent.fetch(parent.orderbook_id, [])
          raise ValidationError, "incomplete combination legs for #{parent.symbol}" unless members.length == parent.number_of_legs && members.length > 1
          raise ValidationError, "duplicate combination leg" unless members.map(&:leg_orderbook_id).uniq.length == members.length
          members.each do |leg|
            child = index[leg.leg_orderbook_id]
            raise ValidationError, "combination leg missing or not outright" unless child && !child.combination? && child.orderbook_id != parent.orderbook_id
          end
          if @code_resolver.is_a?(NamedCodeResolver)
            rule = @code_resolver.naming.rule_for(class_map.fetch(parent.instrument_class_key))
            if rule[:kind] == :calendar_spread
              raise ValidationError, "calendar spread requires two 1:1 legs" unless members.length == 2 && members.all? { |leg| leg.leg_ratio == 1 }
              near = Value.month(parent.metadata.fetch(:contract_month))
              far = Value.month(parent.metadata.fetch(:far_contract_month))
              expected = { near => :sell, far => :buy }
              actual = members.to_h do |leg|
                child = index.fetch(leg.leg_orderbook_id)
                child_rule = @code_resolver.naming.rule_for(class_map.fetch(child.instrument_class_key))
                raise ValidationError, "calendar spread leg product mismatch" unless child.instrument_group == :future && child_rule[:kind] == :future && child_rule[:root] == rule[:root]
                [Value.month(child.metadata.fetch(:contract_month)), leg.leg_side]
              end
              raise ValidationError, "calendar spread must buy far and sell near" unless actual == expected
            end
          end
        end
        legs.each do |leg|
          raise ValidationError, "invalid combination parent" unless index[leg.combo_orderbook_id]&.combination?
        end
      end

      def ensure_unique_current!(instruments)
        identities = Set.new
        codes = Set.new
        instruments.each do |instrument|
          raise ValidationError, "duplicate current economic identity #{instrument.identity_key.inspect}" unless identities.add?(instrument.identity_key)
          next unless instrument.orderbook_id
          raise ValidationError, "duplicate current orderbook_id #{instrument.orderbook_id}" unless codes.add?(instrument.orderbook_id)
        end
      end

      def target_valid?(instrument, target)
        expiry = if @lifecycle_policy
                   @lifecycle_policy.expiry_for(instrument)
                 else
                   instrument.effective_last_trading_date || instrument.last_trading_date
                 end
        (!expiry || expiry >= target) &&
          (instrument.metadata[:availability_date].nil? || Value.date(instrument.metadata[:availability_date]) <= target)
      end

      def current_valid?(instrument, class_map, context)
        klass = class_map[instrument.instrument_class_key]
        klass && klass.tradable? && instrument.active_at?(context, expiry: selected_expiry(instrument)) && target_valid?(instrument, context.target_trading_date)
      end

      def selected_expiry(instrument)
        @lifecycle_policy ? @lifecycle_policy.expiry_for(instrument) : instrument.effective_last_trading_date || instrument.last_trading_date
      end

      def apply_generation_suppression(candidates, overrides, context, audit)
        candidates.reject do |candidate|
          override = overrides.map { |value| Value.keys(value) }.find do |item|
            item[:action].to_s.upcase == "SUPPRESS_GENERATION" &&
              effective_override?(item, context.target_trading_date) && scope_match?(candidate, item)
          end
          next false unless override
          audit << decision("SUPPRESS", candidate, context, override[:reason] || "generation_suppressed")
          true
        end
      end

      def apply_explicit_overrides!(by_identity, overrides, class_map, context, audit)
        overrides.each do |raw|
          override = Value.keys(raw)
          action = override.fetch(:action).to_s.upcase
          if action == "SUPPRESS_GENERATION"
            validate_override_selector!(override, class_map)
            next
          end
          next unless effective_override?(override, context.target_trading_date)
          validate_override_selector!(override, class_map)
          if override[:instrument_class_id]
            klass = class_map.values.find { |k| k.instrument_class_id.to_s == override[:instrument_class_id].to_s }
            override[:instrument_class_key] = klass.instrument_class_key if klass
          end

          case action
          when "FORCE_REMOVE"
            removed = by_identity.values.select { |instrument| scope_match?(instrument, override) }
            removed.each { |instrument| by_identity.delete(instrument.identity_key) }
            removed.each do |instrument|
              event = decision("OVERRIDE_REMOVE", instrument, context, override[:reason] || "official_override")
              event["override_source"] = override[:source]; event["effective_trading_date"] = override[:effective_trading_date]
              audit << event
            end
          when "FORCE_ADD"
            instrument = Instrument.from(override.fetch(:instrument))
            raise ValidationError, "FORCE_ADD instrument disagrees with selector" unless scope_match?(instrument, override)
            if override[:instrument_class_id] && !class_map.key?(instrument.instrument_class_key)
              resolved = class_map.values.find { |klass| klass.instrument_class_id.to_s == override[:instrument_class_id].to_s }
              raise ValidationError, "override instrument class id does not resolve" unless resolved
            end
            raise ValidationError, "override class is outside the maintained scope" unless class_map.key?(instrument.instrument_class_key)
            by_identity[instrument.identity_key] = instrument
            event = decision("OVERRIDE_ADD", instrument, context, override[:reason] || "official_override")
            event["override_source"] = override[:source]; event["effective_trading_date"] = override[:effective_trading_date]
            audit << event
          when "REPLACE_EXPIRY"
            removed = by_identity.values.select do |instrument|
              instrument.last_trading_date == Value.date(override.fetch(:from_expiration_date)) && scope_match?(instrument, override)
            end
            removed.each { |instrument| by_identity.delete(instrument.identity_key) }
            replacement = Instrument.from(override.fetch(:instrument))
            raise ValidationError, "replacement class is outside scope" unless class_map.key?(replacement.instrument_class_key)
            raise ValidationError, "REPLACE_EXPIRY must match exactly one instrument" unless removed.length == 1
            by_identity[replacement.identity_key] = replacement
            removed.each { |instrument| audit << decision("OVERRIDE_REMOVE", instrument, context, "replace_expiry") }
            audit << decision("OVERRIDE_ADD", replacement, context, override[:reason] || "replace_expiry")
          else
            raise ValidationError, "unsupported override action #{action}"
          end
        end
      end

      def effective_override?(override, target)
        !override[:effective_trading_date] || Value.date(override[:effective_trading_date]) == target
      end

      def scope_match?(instrument, override)
        (!override[:instrument_class_key] || instrument.instrument_class_key == override[:instrument_class_key].to_s) &&
          (!override[:instrument_class_id] || instrument.instrument_class_id.nil? || instrument.instrument_class_id.to_s == override[:instrument_class_id].to_s) &&
          (!override[:orderbook_id] || instrument.orderbook_id == Value.orderbook_id(override[:orderbook_id])) &&
          (!override[:instrument_code] || instrument.symbol == override[:instrument_code].to_s) &&
          (!override[:expiration_date] || instrument.last_trading_date == Value.date(override[:expiration_date]))
      end

      def validate_override_selector!(override, class_map)
        raise ValidationError, "unsupported override action" unless %w[FORCE_ADD FORCE_REMOVE REPLACE_EXPIRY SUPPRESS_GENERATION].include?(override[:action].to_s.upcase)
        allowed = %i[action instrument_class_key instrument_class_id orderbook_id instrument_code expiration_date from_expiration_date instrument reason source effective_trading_date]
        unknown = override.keys - allowed
        raise ValidationError, "unknown override selector #{unknown.join(',')}" unless unknown.empty?
        if override[:action].to_s.upcase != "SUPPRESS_GENERATION" && override[:source].to_s.empty?
          raise ValidationError, "override source is required"
        end
        selectors = %i[instrument_class_key instrument_class_id orderbook_id instrument_code expiration_date]
        present = selectors.select { |key| override[key] }
        raise ValidationError, "override selector required" if present.empty?
        if override[:instrument_class_key] && !class_map.key?(override[:instrument_class_key].to_s)
          raise ValidationError, "unknown instrument_class_key selector"
        end
        if override[:instrument_class_id] && !class_map.values.any? { |k| k.instrument_class_id.to_s == override[:instrument_class_id].to_s }
          raise ValidationError, "unknown instrument_class_id selector"
        end
      end

      def validate_final!(instruments, class_map, context)
        identities = Set.new
        codes = Set.new
        symbols = Set.new
        instruments.each do |instrument|
          raise ValidationError, "duplicate economic identity #{instrument.identity_key.inspect}" unless identities.add?(instrument.identity_key)
          if instrument.orderbook_id && !codes.add?(instrument.orderbook_id)
            raise ValidationError, "duplicate orderbook_id #{instrument.orderbook_id}"
          end
          raise ValidationError, "duplicate Symbol #{instrument.symbol}" unless symbols.add?(instrument.symbol)
          next unless class_map[instrument.instrument_class_key]
          raise ValidationError, "non-tradable class in final output" unless class_map[instrument.instrument_class_key].tradable?
          raise ValidationError, "expired instrument in final output" unless target_valid?(instrument, context.target_trading_date)
          raise ValidationError, "inactive instrument in final output" unless instrument.active_at?(context, expiry: selected_expiry(instrument))
          if !context.synthetic?
            id = instrument.orderbook_id
            raise ValidationError, "real OrderBookID must be uint32" unless id.to_s.match?(/\A\d+\z/) && id.to_i.between?(0, 4_294_967_295)
          end
          if @code_resolver.respond_to?(:validate_instrument!)
            @code_resolver.validate_instrument!(instrument, instrument_class: class_map.fetch(instrument.instrument_class_key), context: context)
          end
        end
      end

      def reconcile(actual, official_reference, context)
        return [] unless official_reference
        official = official_reference.map { |item| Instrument.from(item) }
        [actual, official].each do |rows|
          rows.group_by(&:orderbook_id).each { |key, values| raise ValidationError, "ambiguous orderbook #{key}" if key && values.length > 1 }
          rows.group_by { |i| i.symbol || i.orderbook_id || i.identity_key }.each do |key, values|
            raise ValidationError, "ambiguous official duplicate #{key}" if values.length > 1
          end
        end
        ob_index = official.each_with_object({}) { |i, h| h[i.orderbook_id] = i if i.orderbook_id }
        sym_index = official.each_with_object({}) { |i, h| h[i.symbol] = i if i.symbol }
        used = Set.new
        pairs = []
        actual.each do |a|
          by_ob = a.orderbook_id && ob_index[a.orderbook_id]
          by_sym = a.symbol && sym_index[a.symbol]
          if by_ob && by_sym && !by_ob.equal?(by_sym)
            raise ValidationError, "ambiguous cross-identifier reconciliation for #{a.symbol}/#{a.orderbook_id}"
          end
          o = by_ob || by_sym
          if o && used.include?(o.object_id)
            raise ValidationError, "ambiguous official duplicate identity"
          end
          pairs << [a, o]
          used.add(o.object_id) if o
        end
        official.each { |o| pairs << [nil, o] unless used.include?(o.object_id) }
        pairs.map do |actual_item, official_item|
          key = actual_item&.symbol || actual_item&.orderbook_id || official_item&.symbol || official_item&.orderbook_id
          status = if actual_item && official_item
                      diffs = meaningful_diffs(actual_item, official_item)
                      diffs.empty? ? "MATCHED" : "ATTRIBUTE_MISMATCH"
                    elsif actual_item
                      "GENERATED_ONLY"
                    else
                      "OFFICIAL_ONLY"
                    end
          {
            "identity" => key.to_s,
            "status" => status,
            "field_diffs" => (actual_item && official_item ? meaningful_diffs(actual_item, official_item) : nil),
            "target_trading_date" => context.target_trading_date.iso8601
          }
        end
      end

      def meaningful_diffs(actual, official)
        fields = %i[symbol orderbook_id instrument_class_key exchange market commodity_code
                    datetime_first_trading datetime_last_trading number_of_legs vcm_flag isin_code effective_tomorrow
                    last_trading_date effective_last_trading_date strike_price
                    contract_size price_quotation_factor instrument_status modifier instrument_group
                    financial_product put_or_call reference_instrument_status decimal_in_contract_size
                    decimal_in_strike_price decimal_in_price]
        fields.each_with_object({}) do |field, out|
          a = actual.public_send(field)
          o = official.public_send(field)
          out[field.to_s] = { "actual" => (a.respond_to?(:iso8601) ? a.iso8601 : a),
                              "official" => (o.respond_to?(:iso8601) ? o.iso8601 : o) } unless a == o
        end
      end

      def decision(action, instrument, context, reason)
        metadata = instrument.metadata || {}
        {
          "action" => action, "orderbook_id" => instrument.orderbook_id,
          "symbol" => instrument.symbol, "instrument_class_key" => instrument.instrument_class_key,
          "last_trading_date" => instrument.last_trading_date&.iso8601,
          "strike_price" => instrument.strike_price && Value.decimal_string(instrument.strike_price),
          "target_trading_date" => context.target_trading_date.iso8601,
          "source_trading_date" => context.source_trading_date.iso8601,
          "product_profile_id" => metadata[:product_profile_id],
          "source_reference" => metadata[:source_reference],
          "reference_price" => metadata[:reference_price],
          "strike_grid_id" => metadata[:strike_grid_id],
          "coverage_policy" => metadata[:coverage_policy],
          "rule_configuration" => metadata[:rule_configuration],
          "tenor_bucket" => metadata[:tenor_bucket],
          "quote_date" => metadata.dig(:reference_price, :reference_price_date),
          "code_source" => metadata[:code_source],
          "override_source" => metadata[:override_source],
          "reason_code" => reason
        }.compact
      end

      def block_record(klass, context, reason, profile)
        {
          "action" => "BLOCK", "instrument_class_key" => klass.instrument_class_key,
          "product_profile_id" => profile&.id,
          "target_trading_date" => context.target_trading_date.iso8601,
          "reason_code" => reason
        }.compact
      end
    end
  end
end

# Versioned policy configuration is kept in this runtime file so deployments
# do not depend on a sidecar. Values are selected by effective date and an
# unresolved tie is rejected.
module HKEX
module InstrumentMaintenance
module PolicyConfig
  class RuleVersion
    attr_reader :rule_id, :version, :effective_from, :effective_to, :source_reference,
                :evidence_class, :profile_id, :rule_kind, :variant, :value, :priority,
                :approved
    def initialize(rule_id:, version:, source_reference:, evidence_class:, rule_kind:, value:,
                   profile_id: nil, variant: nil, effective_from: nil, effective_to: nil,
                   priority: 0, approved: false, **metadata)
      @rule_id = rule_id.to_s; @version = version.to_s
      @effective_from = effective_from && HKEX::InstrumentMaintenance::Value.date(effective_from)
      @effective_to = effective_to && HKEX::InstrumentMaintenance::Value.date(effective_to)
      @source_reference = source_reference.to_s
      @evidence_class = evidence_class.to_sym; @profile_id = profile_id&.to_s
      @rule_kind = rule_kind.to_sym; @variant = variant&.to_s; @value = value; @priority = Integer(priority)
      @approved = approved == true
      @metadata = metadata
      raise ValidationError, "rule_id, version and source are required" if @rule_id.empty? || @version.empty? || @source_reference.empty?
      raise ValidationError, "invalid rule effective range" if @effective_from && @effective_to && @effective_to < @effective_from
    end
    def effective_on?(date)
      d = HKEX::InstrumentMaintenance::Value.date(date); (!effective_from || d >= effective_from) && (!effective_to || d <= effective_to)
    end
    def selector_match?(profile_id, kind, variant = nil)
      rule_kind == kind.to_sym && (self.profile_id.nil? || self.profile_id == profile_id.to_s) &&
        (self.variant.nil? || self.variant == variant.to_s)
    end
    def specificity; (profile_id ? 1 : 0) + (variant ? 1 : 0) end
    def approved?; approved == true end
  end

  class RuleConfiguration
    attr_reader :entries
    def initialize(entries = []); @entries = entries.map { |e| e.is_a?(RuleVersion) ? e : RuleVersion.new(**e.transform_keys(&:to_sym)) }.freeze end
    def resolve(profile_id:, rule_kind:, on:, variant: nil, rule_id: nil)
      rows = entries.select { |e| e.effective_on?(on) && e.selector_match?(profile_id, rule_kind, variant) && (rule_id.nil? || e.rule_id == rule_id.to_s) }
      return nil if rows.empty?
      rank = rows.map { |e| [e.specificity, e.priority, e.effective_from || Date.new(1, 1, 1)] }.max
      best = rows.select { |e| [e.specificity, e.priority, e.effective_from || Date.new(1, 1, 1)] == rank }
      raise ValidationError, "ambiguous #{rule_kind} configuration" if best.length > 1
      best.first
    end
  end

  class ApprovedStrikeGridConfig
    def initialize(configuration: RuleConfiguration.new); @configuration = configuration end
    def resolve(profile_id:, target_date:, real: true, cross_band: false, variant: nil, rule_id: nil,
                fallback_grid: nil, expected_interval: nil, allowed_range: nil)
      raise ReviewRequired, "cross-band strike grid" if cross_band
      entry = @configuration.resolve(profile_id: profile_id, rule_kind: :strike_grid, on: target_date,
                                     variant: variant, rule_id: rule_id)
      unless entry
        raise ReviewRequired, "approved strike grid required for #{profile_id}/#{variant}" if real
        raise ReviewRequired, "host fallback grid is required" unless fallback_grid.is_a?(ValidStrikeGrid)
        return [fallback_grid, nil]
      end
      unless !real || (entry.approved && %i[hkex_rule official_reference derived_algorithm system_policy].include?(entry.evidence_class))
        raise ReviewRequired, "unapproved strike grid #{entry.rule_id}"
      end
      grid = entry.value.is_a?(ValidStrikeGrid) ? entry.value : ValidStrikeGrid.new(**entry.value.transform_keys(&:to_sym))
      if expected_interval && grid.interval != Value.decimal(expected_interval)
        raise ValidationError, "configured strike interval disagrees with profile"
      end
      if allowed_range
        low, high = allowed_range.map { |value| Value.decimal(value) }
        raise ValidationError, "strike grid anchor outside profile range" unless grid.anchor >= low && grid.anchor <= high
      end
      [grid, entry]
    end
  end

  class LifecyclePolicy
    attr_reader :product_id, :market, :expiry_field, :target_opening_time, :version,
                :source_reference, :evidence_class
    def initialize(product_id:, market:, expiry_field:, target_opening_time:, version:, source_reference:, evidence_class:, approved: false, effective_from: nil, effective_to: nil, **)
      @product_id = product_id.to_s; @market = market.to_s; @expiry_field = expiry_field.to_sym
      @target_opening_time = target_opening_time.to_s; @version = version.to_s
      @source_reference = source_reference.to_s; @evidence_class = evidence_class.to_sym; @approved = approved == true
      @effective_from = effective_from && HKEX::InstrumentMaintenance::Value.date(effective_from)
      @effective_to = effective_to && HKEX::InstrumentMaintenance::Value.date(effective_to)
      raise ValidationError, "lifecycle source and version are required" if @source_reference.empty? || @version.empty?
      raise ValidationError, "invalid lifecycle effective range" if @effective_from && @effective_to && @effective_to < @effective_from
      raise ValidationError, "unsupported lifecycle expiry field" unless %i[last_trading_date effective_last_trading_date].include?(@expiry_field)
    end
    def approved?; @approved end
    def target_opening_at(target_date)
      parsed = Time.parse(@target_opening_time)
      date = HKEX::InstrumentMaintenance::Value.date(target_date)
      Time.new(date.year, date.month, date.day, parsed.hour, parsed.min, parsed.sec, parsed.utc_offset)
    rescue ArgumentError
      raise ValidationError, "invalid target opening time #{@target_opening_time.inspect}"
    end
    def effective_on?(date)
      d = HKEX::InstrumentMaintenance::Value.date(date)
      (!@effective_from || d >= @effective_from) && (!@effective_to || d <= @effective_to)
    end
    def expiry_for(row)
      value = row.respond_to?(:key?) && (row[@expiry_field] || row[@expiry_field.to_s]); value ||= row.public_send(@expiry_field) if row.respond_to?(@expiry_field)
      if value.nil?
        group = row.respond_to?(:instrument_group) ? row.instrument_group : Enums.instrument_group(row[:instrument_group] || row["instrument_group"])
        return nil if %i[exchange_rate payment_currency].include?(group)
        raise ValidationError, "missing lifecycle expiry"
      end
      HKEX::InstrumentMaintenance::Value.date(value)
    end
    def active_at_target?(row, target_date); (expiry = expiry_for(row)).nil? || expiry >= HKEX::InstrumentMaintenance::Value.date(target_date) end
    def project(row); row.dup end
  end
end
end
end

# frozen_string_literal: true

# OMD folder adapter and production runner included in this standalone
# HKEX instrument maintenance file, for consuming a daily folder of OMD CSVs.

require "csv"
require "fileutils"
require "pathname"
require "securerandom"
require "tempfile"

module HKEX
  module InstrumentMaintenance
    ADAPTER_VERSION = "3.0.0"

    class InputError < Error
      attr_reader :source, :file, :row, :code, :details

      def initialize(message, source: nil, file: nil, row: nil, code: :input_error, details: {})
        @source = source&.to_sym
        @file = file
        @row = row
        @code = code.to_sym
        @details = details
        location = [file, row && "row #{row}"].compact.join(":")
        prefix = [source && "source #{source}", location].compact.reject(&:empty?).join(" ")
        super(prefix.empty? ? message : "#{prefix}: #{message}")
      end

      def to_h
        { severity: "ERROR", code: code.to_s, message: message, source: source,
          file: file, row: row, details: details }.compact
      end
    end

    class DiscoveryError < InputError; end
    class HeaderError < InputError; end
    class MappingError < InputError; end
    class DuplicateInputError < InputError; end
    class ForeignKeyError < InputError; end
    class OutputError < Error; end

    Diagnostic = Struct.new(:severity, :code, :message, :source, :file, :row, :details,
                            keyword_init: true) do
      def to_h
        { severity: severity.to_s.upcase, code: code.to_s, message: message,
          source: source&.to_s, file: file, row: row, details: details }.compact
      end
    end

    LocatedRow = Struct.new(:attributes, :source, :file, :row, keyword_init: true) do
      def location
        { source: source, file: file, row: row }
      end
    end

    # Configuration for one OMD entity or reference table. `mapping` follows
    # the feed boundary: CSV header => canonical OMD/domain attribute. `fields`
    # is the inverse form and is useful for constants, lookups and callables.
    class SourceSpec
      ENGINE_ENTITIES = %i[
        underlyings instrument_classes instruments combination_legs market_data
        official_reference overrides reference
      ].freeze
      ENTITY_ALIASES = {
        underlying: :underlyings, instrument_class: :instrument_classes,
        instrument: :instruments, combination_leg: :combination_legs,
        quote: :market_data, quotes: :market_data
      }.freeze

      attr_reader :name, :entity, :patterns, :exclude_patterns, :mapping, :fields,
                  :translations, :types, :required, :defaults, :constants, :key,
                  :encoding, :ignored_headers, :allow_unmapped_headers, :optional,
                  :allow_empty, :metadata, :translation_strict_fields, :encoding_explicit

      def initialize(name, attributes, strict: true)
        h = symbolize_keys(attributes || {})
        @name = name.to_sym
        raw_entity = (h[:entity] || name).to_sym
        @entity = ENTITY_ALIASES.fetch(raw_entity, raw_entity)
        raise ArgumentError, "source #{name}: unknown entity #{raw_entity.inspect}" unless ENGINE_ENTITIES.include?(@entity)
        @patterns = Array(h[:patterns] || h[:pattern]).map(&:to_s).reject(&:empty?).freeze
        @exclude_patterns = Array(h[:exclude_patterns] || h[:exclude]).map(&:to_s).freeze
        @mapping = h.fetch(:mapping, {}).dup.freeze
        @fields = symbolize_keys(h.fetch(:fields, {})).freeze
        @translations = symbolize_keys(h.fetch(:translations, {})).freeze
        @types = symbolize_keys(h.fetch(:types, {})).transform_values { |v| v&.to_sym }.freeze
        @required = Array(h[:required]).map(&:to_sym).freeze
        @defaults = symbolize_keys(h.fetch(:defaults, {})).freeze
        @constants = symbolize_keys(h.fetch(:constants, {})).freeze
        @key = Array(h[:key]).map(&:to_sym).freeze
        @encoding_explicit = h.key?(:encoding) || h.key?(:decoded)
        @encoding = (h[:encoding] || (h[:decoded] == false ? :raw_odp : :decoded_decimal)).to_sym
        @ignored_headers = Array(h[:ignored_headers]).map(&:to_s).freeze
        @allow_unmapped_headers = h.key?(:allow_unmapped_headers) ? !!h[:allow_unmapped_headers] : !strict
        @optional = !!h[:optional]
        @allow_empty = !!h[:allow_empty]
        @metadata = symbolize_keys(h.fetch(:metadata, {})).freeze
        @translation_strict_fields = Array(h[:translation_strict_fields]).map(&:to_sym).freeze
        validate!
      end

      def raw_odp?
        encoding == :raw_odp
      end

      def reference?
        entity == :reference
      end

      private

      def validate!
        raise ArgumentError, "source #{name}: at least one file pattern is required" if patterns.empty?
        unless %i[raw_odp decoded_decimal].include?(encoding)
          raise ArgumentError, "source #{name}: encoding must be raw_odp or decoded_decimal"
        end
        if %i[instrument_classes instruments official_reference].include?(entity) && !encoding_explicit
          raise ArgumentError, "source #{name}: encoding must be explicit for #{entity}"
        end
        invalid = mapping.values.select do |value|
          !(value.is_a?(String) || value.is_a?(Symbol) || value.is_a?(Hash))
        end
        raise ArgumentError, "source #{name}: invalid mapping entry" unless invalid.empty?
        overlap = defaults.keys & constants.keys
        raise ArgumentError, "source #{name}: default and constant overlap: #{overlap.join(', ')}" unless overlap.empty?
      end

      def symbolize_keys(hash)
        hash.each_with_object({}) { |(key, value), out| out[key.to_sym] = value }
      end
    end

    class FolderConfig
      attr_reader :strict, :sources, :foreign_keys, :output, :fail_on_blocked, :verify_unchanged

      def initialize(attributes)
        h = deep_symbolize(attributes || {})
        @strict = h.key?(:strict) ? !!h[:strict] : true
        raw_sources = h.fetch(:sources) { raise ArgumentError, "sources configuration is required" }
        raise ArgumentError, "sources must be a non-empty Hash" unless raw_sources.is_a?(Hash) && !raw_sources.empty?
        @sources = raw_sources.map { |name, value| SourceSpec.new(name, value, strict: strict) }.freeze
        duplicate_names = @sources.group_by(&:name).select { |_, rows| rows.length > 1 }.keys
        raise ArgumentError, "duplicate source names: #{duplicate_names.join(', ')}" unless duplicate_names.empty?
        @foreign_keys = Array(h[:foreign_keys]).map { |fk| normalize_foreign_key(fk) }.freeze
        @output = {
          prefix: h.dig(:output, :prefix),
          overwrite: !!h.dig(:output, :overwrite),
          json: h.dig(:output, :json) != false,
          csv: h.dig(:output, :csv) != false
        }.freeze
        @fail_on_blocked = h.key?(:fail_on_blocked) ? !!h[:fail_on_blocked] : true
        @verify_unchanged = h.key?(:verify_unchanged) ? !!h[:verify_unchanged] : true
      end

      def self.from(value)
        value.is_a?(self) ? value : new(value)
      end

      def source(name)
        sources.find { |item| item.name == name.to_sym }
      end

      private

      def normalize_foreign_key(value)
        h = deep_symbolize(value)
        from = h.fetch(:from).to_sym
        to = h.fetch(:to).to_sym
        fields = Array(h[:fields] || h[:field]).map(&:to_sym)
        references = Array(h[:references] || h[:reference] || fields).map(&:to_sym)
        raise ArgumentError, "foreign key #{from}->#{to} has no fields" if fields.empty?
        raise ArgumentError, "foreign key #{from}->#{to} arity mismatch" unless fields.length == references.length
        { from: from, to: to, fields: fields, references: references,
          required: h.key?(:required) ? !!h[:required] : true,
          allow_blank: !!h[:allow_blank], name: h[:name]&.to_s }
      end

      def deep_symbolize(value)
        case value
        when Hash
          value.each_with_object({}) do |(key, item), out|
            normalized = key.respond_to?(:to_sym) ? key.to_sym : key
            out[normalized] = deep_symbolize(item)
          end
        when Array then value.map { |item| deep_symbolize(item) }
        else value
        end
      end
    end

    class InputBundle
      ENGINE_COLLECTIONS = %i[
        underlyings instrument_classes instruments combination_legs market_data
        official_reference overrides
      ].freeze

      attr_reader(*ENGINE_COLLECTIONS)
      attr_reader :reference_tables, :diagnostics, :manifest, :provenance

      def initialize(collections:, reference_tables:, diagnostics:, manifest:, provenance:)
        ENGINE_COLLECTIONS.each { |name| instance_variable_set("@#{name}", collections.fetch(name, []).freeze) }
        @reference_tables = reference_tables.transform_values(&:freeze).freeze
        @diagnostics = diagnostics.freeze
        @manifest = manifest.freeze
        @provenance = provenance.freeze
        freeze
      end

      def maintenance_arguments
        {
          underlyings: underlyings, instrument_classes: instrument_classes,
          instruments: instruments, combination_legs: combination_legs,
          market_data: market_data, official_reference: official_reference.empty? ? nil : official_reference,
          overrides: overrides
        }
      end

      def counts
        ENGINE_COLLECTIONS.to_h { |name| [name, public_send(name).length] }
          .merge(reference_tables: reference_tables.transform_values(&:length))
      end
    end

    # Loads all configured CSVs before validating cross-file relationships, so
    # file ordering never affects the result.
    class FolderAdapter
      DEFAULT_KEYS = {
        underlyings: %i[commodity_code], instrument_classes: %i[instrument_class_key]
      }.freeze
      FIELD_TYPES = {
        commodity_id: :string, decimal_in_underlying_price: :integer,
        instrument_class_id: :string, contract_size: :decimal,
        price_quotation_factor: :decimal, tick_size: :decimal,
        decimal_in_contract_size: :integer, decimal_in_strike_price: :integer,
        decimal_in_price: :integer, orderbook_id: :orderbook_id,
        last_trading_date: :date, effective_last_trading_date: :date,
        strike_price: :decimal, datetime_first_trading: :time,
        datetime_last_trading: :time, number_of_legs: :integer,
        quote_date: :date, contract_month: :month, value: :decimal,
        combo_orderbook_id: :orderbook_id, leg_orderbook_id: :orderbook_id,
        leg_ratio: :decimal, effective_trading_date: :date,
        ca_created: :boolean, effective_tomorrow: :boolean, is_proxy: :boolean,
        metadata: :json
      }.freeze
      SCALED_ODP_FIELDS = %i[contract_size price_quotation_factor strike_price tick_size].freeze
      CANONICAL_FIELDS = {
        underlyings: %i[commodity_code commodity_id commodity_name underlying_code underlying_type
                        decimal_in_underlying_price base_currency underlying_source ca_created effective_tomorrow],
        instrument_classes: %i[instrument_class_id instrument_class_key instrument_class_name exchange market
                               instrument_group modifier commodity_code contract_size price_quotation_factor tick_size
                               decimal_in_contract_size decimal_in_strike_price decimal_in_price tradable base_currency
                               settlement_currency_id effective_tomorrow generation_mode odp_decoded],
        instruments: %i[orderbook_id symbol instrument_class_id instrument_class_key exchange market instrument_group
                        modifier commodity_code last_trading_date effective_last_trading_date strike_price
                        datetime_first_trading datetime_last_trading instrument_status reference_instrument_status
                        financial_product put_or_call contract_size price_quotation_factor number_of_legs vcm_flag
                        isin_code effective_tomorrow decimal_in_contract_size decimal_in_strike_price decimal_in_price
                        metadata odp_decoded],
        official_reference: %i[orderbook_id symbol instrument_class_id instrument_class_key exchange market instrument_group
                               modifier commodity_code last_trading_date effective_last_trading_date strike_price
                               datetime_first_trading datetime_last_trading instrument_status reference_instrument_status
                               financial_product put_or_call contract_size price_quotation_factor number_of_legs vcm_flag
                               isin_code effective_tomorrow decimal_in_contract_size decimal_in_strike_price decimal_in_price
                               metadata odp_decoded],
        combination_legs: %i[combo_orderbook_id leg_orderbook_id leg_side leg_ratio],
        market_data: %i[underlying_code quote_type quote_date contract_month value source is_proxy],
        overrides: %i[action instrument_class_key instrument_class_id orderbook_id instrument_code expiration_date
                      from_expiration_date instrument reason source effective_trading_date]
      }.freeze

      attr_reader :folder, :config

      def initialize(folder:, config:)
        @folder = File.realpath(File.expand_path(folder.to_s))
        raise ArgumentError, "folder is not a directory: #{@folder}" unless File.directory?(@folder)
        @config = FolderConfig.from(config)
        @diagnostics = []
        @file_digests = {}
        validate_mapping_targets!
      rescue Errno::ENOENT => error
        raise ArgumentError, error.message
      end

      def load
        assignments = discover_files
        source_rows = {}
        manifest_files = []
        provenance = []

        config.sources.each do |spec|
          rows = []
          assignments.fetch(spec.name).each do |path|
            parsed = read_file(spec, path)
            rows.concat(parsed)
            manifest_files << file_manifest(spec, path, parsed.length)
            provenance.concat(parsed.map { |located| located.location.merge(entity: spec.entity) })
          end
          if rows.empty? && !spec.optional && !spec.allow_empty
            raise DiscoveryError.new("no data rows found", source: spec.name, code: :empty_source)
          end
          source_rows[spec.name] = rows
        end

        apply_lookups!(source_rows)
        validate_source_duplicates!(source_rows)
        normalize_omd_rows!(source_rows)
        validate_builtin_relationships!(source_rows)
        validate_configured_foreign_keys!(source_rows)
        validate_domain_rows!(source_rows)

        collections = InputBundle::ENGINE_COLLECTIONS.to_h { |entity| [entity, []] }
        reference_tables = {}
        source_rows.each do |source, rows|
          spec = config.source(source)
          values = rows.map(&:attributes)
          if spec.reference?
            reference_tables[source] = values
          else
            collections.fetch(spec.entity).concat(values)
          end
        end

        validate_entity_duplicates!(collections, source_rows)
        if config.verify_unchanged
          manifest_files.each do |item|
            path = File.join(folder, item.fetch(:path))
            next if File.file?(path) && Digest::SHA256.file(path).hexdigest == item.fetch(:sha256)
            raise InputError.new("file changed before snapshot completed", source: item[:source],
                                 file: item[:path], code: :input_changed)
          end
        end
        manifest = {
          adapter_version: ADAPTER_VERSION, folder: folder,
          configuration_sha256: configuration_digest,
          files: manifest_files.sort_by { |item| [item[:source], item[:path]] },
          counts: collections.transform_values(&:length).merge(reference_tables: reference_tables.transform_values(&:length))
        }
        InputBundle.new(collections: collections, reference_tables: reference_tables,
                        diagnostics: @diagnostics.map(&:to_h), manifest: manifest,
                        provenance: provenance)
      end

      private

      def validate_mapping_targets!
        config.sources.each do |spec|
          next if spec.reference?
          allowed = CANONICAL_FIELDS.fetch(spec.entity)
          targets = spec.mapping.values.map do |definition|
            definition.is_a?(Hash) ? (definition[:field] || definition["field"] || definition[:canonical] || definition["canonical"]) : definition
          end.compact.map(&:to_sym)
          targets.concat(spec.fields.keys)
          targets.concat(spec.defaults.keys)
          targets.concat(spec.constants.keys)
          helper_fields = Array(spec.metadata[:helper_fields]).map(&:to_sym)
          unknown = targets.uniq - allowed - helper_fields
          next if unknown.empty?
          raise ArgumentError, "source #{spec.name}: unknown canonical attributes: #{unknown.join(', ')}"
        end
      end

      def discover_files
        claimed = {}
        config.sources.to_h do |spec|
          files = spec.patterns.flat_map { |pattern| safe_glob(pattern, spec) }.uniq.sort
          excludes = spec.exclude_patterns.flat_map { |pattern| safe_glob(pattern, spec) }.to_set
          files.reject! { |path| excludes.include?(path) }
          if files.empty? && !spec.optional
            raise DiscoveryError.new("no files matched #{spec.patterns.inspect}", source: spec.name,
                                     code: :missing_files)
          end
          files.each do |path|
            if claimed.key?(path)
              raise DiscoveryError.new("file also matched source #{claimed[path]}", source: spec.name,
                                       file: relative(path), code: :overlapping_file_pattern)
            end
            claimed[path] = spec.name
          end
          [spec.name, files]
        end
      end

      def safe_glob(pattern, spec)
        candidate = Pathname.new(pattern)
        if candidate.absolute? || candidate.each_filename.include?("..")
          raise DiscoveryError.new("unsafe file pattern #{pattern.inspect}", source: spec.name,
                                   code: :unsafe_pattern)
        end
        Dir.glob(File.join(folder, pattern), File::FNM_EXTGLOB).select { |path| File.file?(path) }.map do |path|
          real = File.realpath(path)
          unless real == folder || real.start_with?(folder + File::SEPARATOR)
            raise DiscoveryError.new("matched file outside input folder", source: spec.name,
                                     file: path, code: :path_escape)
          end
          real
        end
      end

      def read_file(spec, path)
        before_stat = File.stat(path)
        before_digest = Digest::SHA256.file(path).hexdigest if config.verify_unchanged
        located_rows = []
        File.open(path, "r:bom|utf-8") do |io|
          csv = CSV.new(io, headers: true, return_headers: false, skip_blanks: true)
          headers_checked = false
          csv.each do |csv_row|
            unless headers_checked
              validate_headers!(spec, csv.headers, path)
              headers_checked = true
            end
            line = csv.lineno
            raw = csv_row.to_h.transform_keys { |key| key.to_s.delete_prefix("\uFEFF") }
            next if raw.values.all? { |value| blank?(value) }
            attributes = map_row(spec, raw, path, line)
            attach_provenance!(spec, attributes, path, line)
            located_rows << LocatedRow.new(attributes: attributes.freeze, source: spec.name,
                                           file: relative(path), row: line)
          end
          unless headers_checked
            first_line = io.rewind && io.gets
            if first_line
              validate_headers!(spec, CSV.parse_line(first_line, encoding: "UTF-8"), path)
            end
          end
        end
        after_stat = File.stat(path)
        after_digest = Digest::SHA256.file(path).hexdigest
        if config.verify_unchanged &&
           ([before_stat.dev, before_stat.ino, before_stat.size, before_stat.mtime.to_f, before_digest] !=
            [after_stat.dev, after_stat.ino, after_stat.size, after_stat.mtime.to_f, after_digest])
          raise InputError.new("file changed during import", source: spec.name, file: relative(path),
                               code: :input_changed)
        end
        @file_digests[path] = after_digest
        located_rows
      rescue CSV::MalformedCSVError, EncodingError => error
        raise InputError.new(error.message, source: spec.name, file: relative(path),
                             code: :malformed_csv)
      end

      def validate_headers!(spec, raw_headers, path)
        headers = Array(raw_headers).map { |header| header.to_s.delete_prefix("\uFEFF") }
        duplicate = headers.group_by(&:itself).find { |_, values| values.length > 1 }
        if duplicate
          raise HeaderError.new("duplicate header #{duplicate.first.inspect}", source: spec.name,
                                file: relative(path), code: :duplicate_header)
        end
        exact = spec.mapping.keys.map(&:to_s)
        groups = field_header_groups(spec)
        missing = (exact - headers) + groups.select { |group| (group & headers).empty? }.map { |group| group.join("|") }
        unless missing.empty?
          raise HeaderError.new("missing configured headers: #{missing.join(', ')}", source: spec.name,
                                file: relative(path), code: :missing_headers)
        end
        ambiguous = groups.find { |group| (group & headers).length > 1 }
        if ambiguous
          raise HeaderError.new("multiple aliases present for one field: #{(ambiguous & headers).join(', ')}",
                                source: spec.name, file: relative(path), code: :ambiguous_header_alias)
        end
        expected = (exact + groups.flatten).uniq
        return if spec.allow_unmapped_headers
        unknown = headers - expected - spec.ignored_headers
        unless unknown.empty?
          raise HeaderError.new("unmapped headers: #{unknown.join(', ')}", source: spec.name,
                                file: relative(path), code: :unmapped_headers)
        end
      end

      def field_header_groups(spec)
        spec.fields.values.filter_map do |definition|
          case definition
          when String, Symbol then [definition.to_s]
          when Array then definition.map(&:to_s)
          when Hash
            header = definition[:header] || definition["header"] || definition[:from] || definition["from"]
            Array(header).map(&:to_s) unless header.nil?
          end
        end
      end

      def map_row(spec, raw, path, line)
        out = {}
        per_field = {}
        spec.mapping.each do |header, definition|
          field, options = mapping_definition(definition)
          raise MappingError.new("duplicate mapping for #{field}", source: spec.name,
                                 file: relative(path), row: line, code: :duplicate_mapping) if out.key?(field)
          out[field] = raw[header.to_s]
          per_field[field] = options
        end
        spec.fields.each do |field, definition|
          if definition.is_a?(Hash) && (definition.key?(:lookup) || definition.key?("lookup"))
            out[field] = nil
            per_field[field] = symbolize_hash(definition)
            next
          end
          value, options = evaluate_field(definition, raw, out, spec, path, line)
          out[field] = value
          per_field[field] = options
        end
        spec.constants.each { |field, value| out[field] = callable_value(value, raw, out, spec, path, line) }
        spec.defaults.each { |field, value| out[field] = callable_value(value, raw, out, spec, path, line) if blank?(out[field]) }

        out.each do |field, value|
          options = per_field.fetch(field, {})
          out[field] = normalize_value(spec, field, value, options, raw, path, line)
        end
        missing = spec.required.select { |field| blank?(out[field]) }
        unless missing.empty?
          raise MappingError.new("required attributes are blank: #{missing.join(', ')}", source: spec.name,
                                 file: relative(path), row: line, code: :missing_required_attribute)
        end
        out
      rescue InputError
        raise
      rescue StandardError => error
        raise MappingError.new(error.message, source: spec.name, file: relative(path), row: line,
                               code: :mapping_failed, details: { error_class: error.class.name })
      end

      def mapping_definition(definition)
        return [definition.to_sym, {}] unless definition.is_a?(Hash)
        options = symbolize_hash(definition)
        field = options.delete(:field) || options.delete(:canonical) ||
                raise(ArgumentError, "mapping hash requires field")
        [field.to_sym, options]
      end

      def evaluate_field(definition, raw, current, spec, path, line)
        case definition
        when String, Symbol
          [raw[definition.to_s], {}]
        when Array
          header = definition.map(&:to_s).find { |candidate| raw.key?(candidate) }
          [header && raw[header], {}]
        when Proc
          [invoke(definition, raw, row_context(spec, path, line), current), {}]
        when Hash
          options = symbolize_hash(definition)
          header = options.delete(:header) || options.delete(:from)
          value = if options.key?(:value)
                    callable_value(options.delete(:value), raw, current, spec, path, line)
                  elsif header
                    selected = Array(header).map(&:to_s).find { |candidate| raw.key?(candidate) }
                    selected && raw[selected]
                  elsif options.key?(:default)
                    callable_value(options[:default], raw, current, spec, path, line)
                  end
          [value, options]
        else
          raise ArgumentError, "unsupported field mapping #{definition.inspect}"
        end
      end

      def normalize_value(spec, field, value, options, raw, path, line)
        value = nil if blank?(value)
        translation = options[:translate] || spec.translations[field]
        value = translate(spec, field, value, translation, options, path, line) if translation && !value.nil?
        transform = options[:transform]
        value = invoke(transform, value, raw, row_context(spec, path, line)) if transform
        value = options[:default] if value.nil? && options.key?(:default) && !options[:default].respond_to?(:call)
        required = options[:required] || spec.required.include?(field)
        if required && blank?(value)
          raise MappingError.new("required attribute #{field} is blank", source: spec.name,
                                 file: relative(path), row: line, code: :missing_required_attribute)
        end
        inferred = if spec.raw_odp? && SCALED_ODP_FIELDS.include?(field)
                     nil
                   else
                     FIELD_TYPES[field]
                   end
        coerce(value, options[:type] || spec.types[field] || inferred, spec, field, path, line)
      end

      def translate(spec, field, value, translation, options, path, line)
        definition = translation.is_a?(Hash) ? symbolize_translation_options(translation) : translation
        map = definition.is_a?(Hash) && definition.key?(:map) ? definition[:map] : definition
        strict = options[:translation_strict] || spec.translation_strict_fields.include?(field) ||
                 (definition.is_a?(Hash) && definition[:strict] == true)
        text = value.to_s.strip
        candidates = [value, value.to_s, text, text.upcase, text.downcase]
        candidates << Integer(text, 10) if text.match?(/\A[+-]?\d+\z/)
        candidates << BigDecimal(text) if text.match?(/\A[+-]?(?:\d+(?:\.\d*)?|\.\d+)\z/)
        candidates.uniq!
        key = candidates.find do |candidate|
          map.key?(candidate) || (candidate.respond_to?(:to_sym) && map.key?(candidate.to_sym))
        end
        return map[key] if key && map.key?(key)
        return map[key.to_sym] if key && key.respond_to?(:to_sym) && map.key?(key.to_sym)
        return value unless strict
        raise MappingError.new("unmapped value #{value.inspect} for #{field}", source: spec.name,
                               file: relative(path), row: line, code: :unmapped_value)
      end

      def symbolize_translation_options(value)
        return value unless value.key?(:map) || value.key?("map") || value.key?(:strict) || value.key?("strict")
        { map: value[:map] || value["map"] || {}, strict: value[:strict] || value["strict"] }
      end

      def coerce(value, type, spec, field, path, line)
        return nil if value.nil?
        return value if type.nil?
        case type.to_sym
        when :string then value.to_s.strip
        when :integer then Integer(value.to_s, 10)
        when :decimal then Value.decimal(value)
        when :date then Value.date(value)
        when :month then Value.month(value)
        when :time then Value.time(value)
        when :boolean then Enums.boolean(value, field.to_s)
        when :symbol then value.to_s.strip.downcase.to_sym
        when :orderbook_id then Value.orderbook_id(value.to_s.strip)
        when :json
          value.is_a?(String) ? JSON.parse(value, symbolize_names: true) : value
        else
          raise ArgumentError, "unsupported type #{type.inspect}"
        end
      rescue ArgumentError, JSON::ParserError, ValidationError => error
        raise MappingError.new("cannot coerce #{field}=#{value.inspect} as #{type}: #{error.message}",
                               source: spec.name, file: relative(path), row: line,
                               code: :type_coercion)
      end

      def attach_provenance!(spec, attributes, path, line)
        return unless %i[instruments official_reference].include?(spec.entity)
        metadata = attributes[:metadata].is_a?(Hash) ? attributes[:metadata].dup : {}
        spec.metadata.each { |key, value| metadata[key] = value unless metadata.key?(key) }
        metadata[:adapter_provenance] = { source: spec.name.to_s, file: relative(path), row: line }
        attributes[:metadata] = metadata
      end

      def normalize_omd_rows!(source_rows)
        all = source_rows.values.flatten
        class_rows = all.select { |row| config.source(row.source).entity == :instrument_classes }
        class_rows.each do |row|
          require_explicit_generation_mode!(row)
          decode_located_odp!(row)
        end
        classes = class_rows.group_by { |row| row.attributes[:instrument_class_key].to_s }
        duplicate = classes.find { |key, rows| !key.empty? && rows.length > 1 }
        if duplicate
          raise DuplicateInputError.new("duplicate InstrumentClassKey #{duplicate.first.inspect}",
                                        **duplicate.last.last.location, code: :duplicate_key)
        end
        instrument_rows = all.select do |row|
          %i[instruments official_reference].include?(config.source(row.source).entity)
        end
        instrument_rows.each do |row|
          key = row.attributes[:instrument_class_key].to_s
          klass = classes[key]&.first
          unless klass
            raise ForeignKeyError.new("missing instrument class for #{key.inspect}", **row.location,
                                      code: :foreign_key_missing)
          end
          attrs = row.attributes.dup
          %i[decimal_in_contract_size decimal_in_strike_price decimal_in_price].each do |field|
            attrs[field] = klass.attributes[field] if blank?(attrs[field])
          end
          row.attributes = attrs.freeze
          decode_located_odp!(row)
          inherit_and_validate_class_fields!(row, klass)
        end
      end

      def require_explicit_generation_mode!(row)
        return unless blank?(row.attributes[:generation_mode])
        raise MappingError.new(
          "generation_mode is required; map it or configure an explicit default (usually MAINTAIN_EXISTING_ONLY)",
          **row.location, code: :unsafe_generation_mode
        )
      end

      def decode_located_odp!(row)
        spec = config.source(row.source)
        return unless spec.raw_odp?
        row.attributes = ODPAdapter.decode(row.attributes, decoded: false).freeze
      rescue ValidationError => error
        raise MappingError.new(error.message, **row.location, code: :odp_decode)
      end

      def inherit_and_validate_class_fields!(instrument_row, class_row)
        inherited = %i[instrument_class_id exchange market instrument_group modifier commodity_code
                       contract_size price_quotation_factor decimal_in_contract_size
                       decimal_in_strike_price decimal_in_price]
        attrs = instrument_row.attributes.dup
        inherited.each do |field|
          class_value = class_row.attributes[field]
          next if class_value.nil?
          if blank?(attrs[field])
            attrs[field] = class_value
          elsif !canonical_equal?(field, attrs[field], class_value)
            raise MappingError.new(
              "instrument #{field}=#{attrs[field].inspect} conflicts with class value #{class_value.inspect}",
              **instrument_row.location, code: :class_attribute_conflict,
              details: { field: field, instrument_class_key: attrs[:instrument_class_key] }
            )
          end
        end
        instrument_row.attributes = attrs.freeze
      end

      def canonical_equal?(field, left, right)
        case field
        when :instrument_group then Enums.instrument_group(left) == Enums.instrument_group(right)
        when :contract_size, :price_quotation_factor then Value.decimal(left) == Value.decimal(right)
        else left.to_s == right.to_s
        end
      rescue ValidationError, ArgumentError
        false
      end

      def validate_domain_rows!(source_rows)
        constructors = {
          underlyings: ->(attrs) { Underlying.new(attrs) },
          instrument_classes: ->(attrs) { InstrumentClass.new(attrs) },
          instruments: ->(attrs) { Instrument.new(attrs) },
          official_reference: ->(attrs) { Instrument.new(attrs) },
          combination_legs: ->(attrs) { CombinationLeg.new(attrs) },
          market_data: ->(attrs) { MarketReferenceData::Quote.new(attrs) }
        }
        source_rows.each do |source, rows|
          constructor = constructors[config.source(source).entity]
          next unless constructor
          rows.each do |row|
            constructor.call(row.attributes)
          rescue KeyError, ArgumentError, ValidationError => error
            raise MappingError.new(error.message, **row.location, code: :domain_validation,
                                   details: { error_class: error.class.name })
          end
        end
        validate_instrument_semantics!(source_rows)
      end

      def validate_instrument_semantics!(source_rows)
        all = source_rows.values.flatten
        class_rows = all.select { |row| config.source(row.source).entity == :instrument_classes }
        classes = class_rows.to_h do |row|
          [row.attributes[:instrument_class_key].to_s, InstrumentClass.new(row.attributes)]
        end
        all.each do |row|
          next unless %i[instruments official_reference].include?(config.source(row.source).entity)
          instrument = Instrument.new(row.attributes)
          klass = classes.fetch(instrument.instrument_class_key)
          if klass.option?
            raise ValidationError, "instrument FinancialProduct disagrees with option class" unless instrument.financial_product == :option
            raise ValidationError, "option requires positive strike and LTD" unless instrument.strike_price&.positive? && instrument.last_trading_date
          elsif klass.future?
            raise ValidationError, "instrument FinancialProduct disagrees with future class" unless instrument.financial_product == :future
            raise ValidationError, "future requires LTD and zero/absent strike" unless instrument.last_trading_date && (instrument.strike_price.nil? || instrument.strike_price.zero?)
          end
          expected = case klass.instrument_group
                     when :call_american, :call_european then :call
                     when :put_american, :put_european then :put
                     else :undefined
                     end
          raise ValidationError, "instrument PutOrCall disagrees with class" unless instrument.put_or_call == expected
        rescue KeyError, ArgumentError, ValidationError => error
          raise MappingError.new(error.message, **row.location, code: :domain_relationship,
                                 details: { error_class: error.class.name })
        end
      end

      def apply_lookups!(source_rows)
        config.sources.each do |spec|
          lookups = spec.fields.select do |_, definition|
            definition.is_a?(Hash) && (definition.key?(:lookup) || definition.key?("lookup"))
          end
          next if lookups.empty?
          plans = lookups.map do |field, definition|
            options = symbolize_hash(definition)
            lookup = symbolize_hash(options.fetch(:lookup))
            target_name = lookup.fetch(:source).to_sym
            target_rows = source_rows.fetch(target_name) do
              raise MappingError.new("unknown lookup source #{target_name}", source: spec.name,
                                     code: :unknown_lookup_source)
            end
            local_fields = Array(lookup[:local] || lookup[:fields]).map(&:to_sym)
            foreign_fields = Array(lookup[:foreign] || lookup[:references] || local_fields).map(&:to_sym)
            unless local_fields.length == foreign_fields.length && !local_fields.empty?
              raise MappingError.new("lookup arity mismatch", source: spec.name, code: :lookup_arity)
            end
            index = target_rows.group_by do |target|
              foreign_fields.map { |key| target.attributes[key]&.to_s }
            end
            [field, target_name, local_fields, (lookup[:value] || lookup[:field] || field).to_sym,
             lookup.key?(:required) ? !!lookup[:required] : true, index]
          end
          source_rows.fetch(spec.name).each do |located|
            attributes = located.attributes.dup
            plans.each do |field, target_name, local_fields, target_field, required, index|
              local_key = local_fields.map { |key| attributes[key]&.to_s }
              matches = index.fetch(local_key, [])
              if matches.length > 1
                raise MappingError.new("ambiguous lookup #{target_name} for #{local_key.inspect}",
                                       **located.location, code: :ambiguous_lookup)
              elsif matches.empty?
                if required
                  raise MappingError.new("lookup #{target_name} not found for #{local_key.inspect}",
                                         **located.location, code: :lookup_not_found)
                end
                warning(:lookup_not_found, "optional lookup #{target_name} not found", located,
                        key: local_key)
                next
              end
              attributes[field] = matches.first.attributes[target_field]
            end
            located.attributes = attributes.freeze
          end
        end
      end

      def validate_source_duplicates!(source_rows)
        config.sources.each do |spec|
          keys = spec.key
          next if keys.empty?
          validate_unique!(source_rows.fetch(spec.name), keys, "source #{spec.name}")
        end
      end

      def validate_entity_duplicates!(collections, source_rows)
        DEFAULT_KEYS.each do |entity, keys|
          rows = source_rows.values.flatten.select { |row| config.source(row.source).entity == entity }
          validate_unique!(rows, keys, "entity #{entity}")
        end
        instruments = source_rows.values.flatten.select { |row| config.source(row.source).entity == :instruments }
        validate_unique!(instruments.select { |row| !blank?(row.attributes[:orderbook_id]) }, [:orderbook_id], "entity instruments")
        validate_unique!(instruments.select { |row| !blank?(row.attributes[:symbol]) }, [:symbol], "entity instruments")
        official = source_rows.values.flatten.select { |row| config.source(row.source).entity == :official_reference }
        validate_unique!(official.select { |row| !blank?(row.attributes[:orderbook_id]) }, [:orderbook_id], "entity official_reference")
        validate_unique!(official.select { |row| !blank?(row.attributes[:symbol]) }, [:symbol], "entity official_reference")
      end

      def validate_unique!(rows, keys, label)
        seen = {}
        rows.each do |located|
          key = keys.map { |field| located.attributes[field]&.to_s }
          if key.any?(&:nil?) || key.any?(&:empty?)
            raise DuplicateInputError.new("blank key #{keys.inspect}", **located.location,
                                          code: :blank_key)
          end
          previous = seen[key]
          if previous
            raise DuplicateInputError.new("duplicate #{label} key #{key.inspect}; first at #{previous.file}:#{previous.row}",
                                          **located.location, code: :duplicate_key)
          end
          seen[key] = located
        end
      end

      def validate_builtin_relationships!(source_rows)
        rows = source_rows.values.flatten
        by_entity = rows.group_by { |row| config.source(row.source).entity }
        underlyings = by_entity.fetch(:underlyings, [])
        classes = by_entity.fetch(:instrument_classes, [])
        instruments = by_entity.fetch(:instruments, [])
        legs = by_entity.fetch(:combination_legs, [])
        quotes = by_entity.fetch(:market_data, [])
        validate_fk_rows!(classes, [:commodity_code], underlyings, [:commodity_code], true, "class underlying") unless classes.empty?
        validate_fk_rows!(instruments, [:instrument_class_key], classes, [:instrument_class_key], true, "instrument class") unless instruments.empty?
        # Several commodities (for example HSI and MHI) may share one economic
        # underlying code, so quotes require membership rather than uniqueness.
        validate_fk_rows!(quotes, [:underlying_code], underlyings, [:underlying_code], true,
                          "quote underlying", false, allow_duplicate_reference: true) unless quotes.empty?
        unless legs.empty?
          validate_fk_rows!(legs, [:combo_orderbook_id], instruments, [:orderbook_id], true, "combination parent")
          validate_fk_rows!(legs, [:leg_orderbook_id], instruments, [:orderbook_id], true, "combination leg")
        end
      end

      def validate_configured_foreign_keys!(source_rows)
        config.foreign_keys.each do |fk|
          from = rows_for(fk[:from], source_rows)
          to = rows_for(fk[:to], source_rows)
          validate_fk_rows!(from, fk[:fields], to, fk[:references], fk[:required],
                            fk[:name] || "#{fk[:from]} -> #{fk[:to]}", fk[:allow_blank])
        end
      end

      def rows_for(name, source_rows)
        return source_rows.fetch(name) if source_rows.key?(name)
        matches = source_rows.values.flatten.select { |row| config.source(row.source).entity == name.to_sym }
        raise ArgumentError, "unknown source or entity #{name}" if matches.empty?
        matches
      end

      def validate_fk_rows!(from_rows, fields, to_rows, references, required, label, allow_blank = false,
                            allow_duplicate_reference: false)
        index = to_rows.group_by { |row| references.map { |field| row.attributes[field]&.to_s } }
        ambiguous = index.find { |_, values| values.length > 1 } unless allow_duplicate_reference
        if ambiguous
          location = ambiguous.last.last.location
          raise ForeignKeyError.new("ambiguous reference #{label} key #{ambiguous.first.inspect}",
                                    **location, code: :ambiguous_reference)
        end
        from_rows.each do |row|
          key = fields.map { |field| row.attributes[field]&.to_s }
          next if allow_blank && key.all? { |value| blank?(value) }
          next if index.key?(key)
          if required
            raise ForeignKeyError.new("missing reference #{label} for #{key.inspect}",
                                      **row.location, code: :foreign_key_missing)
          end
          warning(:foreign_key_missing, "optional reference #{label} is missing", row, key: key)
        end
      end

      def file_manifest(spec, path, count)
        { source: spec.name.to_s, entity: spec.entity.to_s, path: relative(path),
          bytes: File.size(path), sha256: @file_digests.fetch(path) { Digest::SHA256.file(path).hexdigest },
          row_count: count, encoding: spec.encoding.to_s }
      end

      def configuration_digest
        descriptor = config.sources.map do |spec|
          {
            name: spec.name, entity: spec.entity, patterns: spec.patterns,
            exclude_patterns: spec.exclude_patterns,
            mapping: stable_config_value(spec.mapping), fields: stable_config_value(spec.fields),
            translations: stable_config_value(spec.translations), types: spec.types,
            required: spec.required, defaults: stable_config_value(spec.defaults),
            constants: stable_config_value(spec.constants), key: spec.key,
            encoding: spec.encoding, allow_unmapped_headers: spec.allow_unmapped_headers,
            ignored_headers: spec.ignored_headers, optional: spec.optional,
            allow_empty: spec.allow_empty, metadata: stable_config_value(spec.metadata),
            translation_strict_fields: spec.translation_strict_fields
          }
        end
        complete = { strict: config.strict, sources: descriptor,
                     foreign_keys: stable_config_value(config.foreign_keys),
                     output: stable_config_value(config.output),
                     fail_on_blocked: config.fail_on_blocked,
                     verify_unchanged: config.verify_unchanged }
        Digest::SHA256.hexdigest(JSON.generate(json_config_value(complete)))
      end

      def stable_config_value(value)
        case value
        when Hash
          value.keys.sort_by(&:to_s).to_h { |key| [key, stable_config_value(value[key])] }
        when Array then value.map { |item| stable_config_value(item) }
        when Proc then "proc@#{value.source_location&.join(':') || 'runtime'}"
        when Symbol then value.to_s
        else value
        end
      end

      def json_config_value(value)
        case value
        when Hash then value.to_h { |key, item| [key.to_s, json_config_value(item)] }
        when Array then value.map { |item| json_config_value(item) }
        when BigDecimal then Value.decimal_string(value)
        else value
        end
      end

      def row_context(spec, path, line)
        { source: spec.name, file: relative(path), row: line }
      end

      def callable_value(value, raw, current, spec, path, line)
        value.respond_to?(:call) ? invoke(value, raw, row_context(spec, path, line), current) : value
      end

      def invoke(callable, *arguments)
        return callable unless callable.respond_to?(:call)
        arity = callable.arity
        arity.negative? ? callable.call(*arguments) : callable.call(*arguments.first(arity))
      end

      def warning(code, message, located, details = {})
        @diagnostics << Diagnostic.new(severity: :warning, code: code, message: message,
                                       **located.location, details: details)
      end

      def relative(path)
        Pathname.new(path).relative_path_from(Pathname.new(folder)).to_s
      end

      def blank?(value)
        value.nil? || (value.respond_to?(:empty?) && value.empty?) ||
          (value.is_a?(String) && value.strip.empty?)
      end

      def symbolize_hash(value)
        value.each_with_object({}) { |(key, item), out| out[key.to_sym] = item }
      end
    end

    class FolderRunResult
      attr_reader :bundle, :maintenance_result, :output_files, :diagnostics, :run_directory

      def initialize(bundle:, maintenance_result:, output_files:, diagnostics:, run_directory: nil)
        @bundle = bundle
        @maintenance_result = maintenance_result
        @output_files = output_files.freeze
        @diagnostics = diagnostics.freeze
        @run_directory = run_directory
      end

      def instruments
        maintenance_result.instruments
      end

      def to_h
        {
          adapter_version: ADAPTER_VERSION, input: bundle.manifest,
          counts: bundle.counts.merge(output_instruments: instruments.length),
          output_files: output_files, diagnostics: diagnostics,
          result: maintenance_result.full_model
        }
      end
    end

    # Orchestrates load -> generate -> merge -> reconcile. The default engine
    # uses global_merge because a folder represents the complete current OMD
    # snapshot; production deployments should inject their approved calendar,
    # rules and official code resolver.
    class FolderRunner
      attr_reader :folder, :config, :engine

      def initialize(folder:, config:, engine: nil)
        @folder = File.expand_path(folder.to_s)
        @config = FolderConfig.from(config)
        @engine = engine || MaintenanceEngine.new(scope_mode: :global_merge)
      end

      def run(context:, output_dir: nil, write: !output_dir.nil?, mode: :standard,
              target_underlying_ids: nil, overrides: nil, official_reference: nil,
              fail_on_blocked: config.fail_on_blocked)
        bundle = FolderAdapter.new(folder: folder, config: config).load
        if bundle.underlyings.empty? || bundle.instrument_classes.empty?
          raise InputError.new("runner requires non-empty Underlying and InstrumentClass sources",
                               code: :incomplete_snapshot)
        end
        arguments = bundle.maintenance_arguments.merge(
          context: context, mode: mode, target_underlying_ids: target_underlying_ids
        )
        arguments[:overrides] = overrides unless overrides.nil?
        arguments[:official_reference] = official_reference unless official_reference.nil?
        result = engine.maintain(**arguments)
        if fail_on_blocked && !result.blocked.empty?
          raise GenerationBlocked, result.blocked.map { |row| row["reason_code"] }.join(", ")
        end
        diagnostics = bundle.diagnostics + result.blocked.map do |row|
          Diagnostic.new(severity: :warning, code: :generation_blocked,
                         message: row["reason_code"], details: row).to_h
        end
        output_files = {}
        run_directory = nil
        if write
          raise ArgumentError, "output_dir is required when write is true" if output_dir.nil?
          input_root = File.realpath(folder)
          expanded_output = resolved_output_path(output_dir)
          if expanded_output == input_root || expanded_output.start_with?(input_root + File::SEPARATOR)
            raise OutputError, "output_dir must be outside the input folder"
          end
          run_directory, output_files = OutputWriter.new(config: config).write(
            output_dir: output_dir, context: context, bundle: bundle,
            maintenance_result: result, diagnostics: diagnostics, mode: mode
          )
        end
        FolderRunResult.new(bundle: bundle, maintenance_result: result,
                            output_files: output_files, diagnostics: diagnostics,
                            run_directory: run_directory)
      end

      private

      # Resolve existing parent symlinks even when the output directory itself
      # does not exist yet (for example, macOS /var -> /private/var).
      def resolved_output_path(path)
        expanded = File.expand_path(path.to_s)
        existing = expanded
        existing = File.dirname(existing) until File.exist?(existing)
        File.expand_path(File.join(File.realpath(existing), expanded.delete_prefix(existing).delete_prefix(File::SEPARATOR)))
      end
    end

    class OutputWriter
      INSTRUMENT_COLUMNS = %w[
        orderbook_id symbol instrument_class_id instrument_class_key exchange market
        instrument_group modifier commodity_code last_trading_date
        effective_last_trading_date strike_price datetime_first_trading
        datetime_last_trading instrument_status reference_instrument_status
        financial_product put_or_call contract_size price_quotation_factor
        number_of_legs vcm_flag isin_code effective_tomorrow
        decimal_in_contract_size decimal_in_strike_price decimal_in_price metadata
      ].freeze

      def initialize(config:)
        @config = config
      end

      def write(output_dir:, context:, bundle:, maintenance_result:, diagnostics:, mode:)
        root = File.expand_path(output_dir.to_s)
        FileUtils.mkdir_p(root)
        raise OutputError, "output path is not a directory" unless File.directory?(root)
        prefix = output_prefix(context)
        final_dir = safe_child(root, prefix)
        stage = safe_child(root, ".#{prefix}.tmp-#{Process.pid}-#{SecureRandom.hex(6)}")
        FileUtils.mkdir(stage, mode: 0o750)
        files = {}
        begin
          payloads = build_payloads(bundle, maintenance_result, diagnostics, context, mode)
          payloads.each do |name, content|
            next if name.end_with?(".json") && !@config.output[:json]
            next if name.end_with?(".csv") && !@config.output[:csv]
            path = File.join(stage, name)
            atomic_file(path, content)
            files[name] = File.join(final_dir, name)
          end
          publish(stage, final_dir)
        rescue StandardError
          FileUtils.rm_rf(stage) if File.exist?(stage)
          raise
        end
        [final_dir, files.transform_keys { |name| File.basename(name, File.extname(name)).to_sym }]
      end

      private

      def output_prefix(context)
        trading_date = context.target_trading_date || context.source_trading_date
        configured = @config.output[:prefix]
        value = if configured.respond_to?(:call)
                  configured.call(context)
                elsif configured
                  configured.to_s.gsub("%Y%m%d", trading_date.strftime("%Y%m%d"))
                else
                  "hkex_maintenance_#{trading_date.strftime('%Y%m%d')}"
                end
        unless value.match?(/\A[A-Za-z0-9][A-Za-z0-9_.-]*\z/)
          raise OutputError, "unsafe output prefix #{value.inspect}"
        end
        value
      end

      def build_payloads(bundle, result, diagnostics, context, mode)
        sorted = result.instruments.sort_by do |item|
          [item.instrument_class_key.to_s, item.last_trading_date || Date.new(9999, 12, 31),
           item.strike_price || BigDecimal("0"), item.put_or_call.to_s, item.symbol.to_s]
        end
        model = result.full_model.merge(
          "adapter_version" => ADAPTER_VERSION,
          "input_manifest" => json_ready(bundle.manifest),
          "adapter_diagnostics" => json_ready(diagnostics)
        )
        {
          "result.json" => JSON.pretty_generate(json_ready(model)) + "\n",
          "instruments.csv" => instruments_csv(sorted),
          "audit.csv" => hashes_csv(result.audit),
          "reconciliation.csv" => hashes_csv(result.reconciliation),
          "manifest.json" => JSON.pretty_generate(json_ready(bundle.manifest.merge(
            source_trading_date: context.source_trading_date,
            target_trading_date: context.target_trading_date,
            as_of_timestamp: context.as_of_timestamp,
            calendar_mode: context.calendar_mode,
            maintenance_mode: mode,
            output_instrument_count: sorted.length,
            blocked_count: result.blocked.length,
            audit_count: result.audit.length,
            reconciliation_count: result.reconciliation.length,
            audit_actions: result.audit.group_by { |row| row["action"] }.transform_values(&:length),
            reconciliation_statuses: result.reconciliation.group_by { |row| row["status"] }.transform_values(&:length)
          ))) + "\n",
          "diagnostics.json" => JSON.pretty_generate(json_ready(diagnostics)) + "\n"
        }
      end

      def instruments_csv(instruments)
        CSV.generate do |csv|
          csv << INSTRUMENT_COLUMNS
          instruments.each do |instrument|
            row = instrument.to_h
            csv << INSTRUMENT_COLUMNS.map { |field| csv_value(row[field] || row[field.to_sym]) }
          end
        end
      end

      def hashes_csv(rows)
        flat = rows.map { |row| flatten_hash(json_ready(row)) }
        headers = flat.flat_map(&:keys).uniq.sort
        CSV.generate do |csv|
          csv << headers
          flat.each { |row| csv << headers.map { |header| csv_value(row[header]) } }
        end
      end

      def flatten_hash(value, prefix = nil, out = {})
        value.each do |key, item|
          name = [prefix, key].compact.join(".")
          if item.is_a?(Hash)
            flatten_hash(item, name, out)
          else
            out[name] = item
          end
        end
        out
      end

      def csv_value(value)
        case value
        when Hash, Array then JSON.generate(json_ready(value))
        when Date, Time, DateTime then value.iso8601
        when BigDecimal then Value.decimal_string(value)
        when Symbol then value.to_s
        else value
        end
      end

      def json_ready(value)
        case value
        when Hash then value.each_with_object({}) { |(key, item), out| out[key.to_s] = json_ready(item) }
        when Array then value.map { |item| json_ready(item) }
        when Date, Time, DateTime then value.iso8601
        when BigDecimal then Value.decimal_string(value)
        when Symbol then value.to_s
        else value
        end
      end

      def atomic_file(path, content)
        Tempfile.create([".write-", ".tmp"], File.dirname(path), mode: File::RDWR, encoding: "UTF-8") do |file|
          file.write(content)
          file.flush
          file.fsync
          File.chmod(0o640, file.path)
          File.rename(file.path, path)
        end
      end

      def publish(stage, final_dir)
        if File.exist?(final_dir)
          raise OutputError, "output run already exists: #{final_dir}" unless @config.output[:overwrite]
          backup = "#{final_dir}.backup-#{SecureRandom.hex(6)}"
          File.rename(final_dir, backup)
          begin
            File.rename(stage, final_dir)
            FileUtils.rm_rf(backup)
          rescue StandardError
            File.rename(backup, final_dir) unless File.exist?(final_dir)
            raise
          end
        else
          File.rename(stage, final_dir)
        end
      end

      def safe_child(root, name)
        child = File.expand_path(name, root)
        unless child.start_with?(root + File::SEPARATOR)
          raise OutputError, "output path escapes output directory"
        end
        child
      end
    end

    # Descriptive aliases retained for callers that use snapshot/maintenance
    # terminology in deployment configuration.
    FolderSnapshot = InputBundle
    FolderMaintenanceRunner = FolderRunner
  end
end
